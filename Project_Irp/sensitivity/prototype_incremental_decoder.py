"""Step 1 prototype (sequential, no multiprocessing, ~20-30 generations):
does the incremental decoder (see diagnose_incremental_decoder_continuity.py
-- edit the parent's already-built route by a bounded number of local moves
instead of reconstructing from scratch) work correctly and show a sane
quality trend when wired into a REAL search loop, before investing in a full
parallelised campaign?

Each individual now carries its own route_result and priorities across
generations (route_pool / priority_pool, indexed by population SLOT --
same "no stable individual identity across elitist survival" approximation
already used and accepted in this project for pbest/velocity/lambda in
remedies C/D/E, tracked here via a custom pymoo Population attribute so the
right route follows the right survivor, not just a slot-position guess).

Per (period, frigo/nonfrigo group), the incremental path is used ONLY if
BOTH hold, else falls back to the exact production `_nearest_neighbour`
reconstruction for that group:
  1. the set of served clients is IDENTICAL to last time (no client's
     demand/eligibility genuinely changed) -- relocation cannot add or
     remove a client, so a set change is unrepresentable incrementally;
  2. every truck's new total load (same clients, new quantities) still
     fits its capacity -- a quantity change can bust a partition that used
     to fit.
When eligible, quantities are updated in place (order untouched) and up to
K_MAX*alpha/alpha_max clients (globally, whichever priority moved most)
are relocated to their cheapest feasible position within their OWN existing
route via `_relocate_client` (reuses repair.py's already-validated
`_evaluate_candidate`, decoder-agnostic, checks tau_max feasibility itself).

Isolation: `Solvers/NSGA3/decoder.py` and `Solvers/QINSGA3/repair.py` are
NEVER modified -- only their existing pure/private helpers are reused.

Usage:
    python -m sensitivity.prototype_incremental_decoder --gen 30 --seed 42
"""
from __future__ import annotations

import argparse
import os
import sys
import time

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from models.parametres          import load_instance
from Solvers.NSGA3.decoder       import (
    decode_chromosome, build_routes, _nearest_neighbour,
    _force_min_release, _clamp_to_integer_budget,
)
from Solvers.NSGA3.evaluator     import compute_f1, compute_f2, compute_f3, compute_f4
from Solvers.NSGA3.problem       import IRPProblem
from Solvers.QINSGA3.algorithm   import (
    _encode_theta, _penalised_F, _normalise_F, _assign_ref_dirs,
    _select_guides, _supplement_from_archive, _migrate, _crowding_trim,
    _archive_update, _build_g_constraints,
)
from Solvers.QINSGA3.chromosome  import QuantumPopulation
from Solvers.QINSGA3.repair      import (
    _route_traversal_time, _repair_route_result,
)
from sensitivity.diagnose_incremental_decoder_continuity import _relocate_client

N_PARTITIONS = 8
N_OBJ        = 4
K_MAX        = 20


# ---------------------------------------------------------------------------
# Incremental build_routes -- hybrid: incremental edit when safe, exact
# production _nearest_neighbour fallback otherwise
# ---------------------------------------------------------------------------

def _build_routes_incremental(quantities, sets_, params_, priorities,
                               prev_route_result, prev_priorities, k_moves, stats):
    """Same per-period/frigo-nonfrigo structure as decoder.py::build_routes
    (copied, not imported -- build_routes doesn't expose the per-group step
    standalone). For each (t, group), tries the incremental path first;
    falls back to the exact `_nearest_neighbour` production call whenever
    the incremental preconditions (see module docstring) aren't met. `stats`
    (a dict with int counters) is mutated to record how often each path is
    taken -- purely for this prototype's own reporting, not used by the
    algorithm itself.
    """
    clients          = sets_["clients"]
    T                = sets_["T"]
    M                = sets_["M"]
    O                = sets_["O"]
    Q                = params_["Q"]
    d                = params_["d"]
    v                = params_["v"]
    s                = params_["s"]
    requires_cold    = params_["requires_cold"]
    frigo_trucks     = params_["frigo_trucks"]
    non_frigo_trucks = [k for k in M if k not in frigo_trucks]
    frigo_list       = sorted(frigo_trucks)
    q_lt             = params_["q_lt"]
    min_delivery     = params_.get("min_delivery_threshold", 5)

    I_frigo    = float(params_["I_O_init_frigo"])
    I_nonfrigo = float(params_["I_O_init_nonfrigo"])
    I_min_f    = params_["I_O_min_frigo"]
    I_max_f    = params_["I_O_max_frigo"]
    I_min_nf   = params_["I_O_min_nonfrigo"]
    I_max_nf   = params_["I_O_max_nonfrigo"]
    R_frigo    = params_["R_frigo"]
    R_nonfrigo = params_["R_nonfrigo"]

    frigo_total    = {l: sum(q_lt[l, t] for t in T if     requires_cold[l, t]) for l in clients}
    nonfrigo_total = {l: sum(q_lt[l, t] for t in T if not requires_cold[l, t]) for l in clients}

    cum_frigo_demand = {
        (l, t): sum(q_lt[l, td] for td in T if td <= t and     requires_cold[l, td])
        for l in clients for t in T
    }
    cum_nonfrigo_demand = {
        (l, t): sum(q_lt[l, td] for td in T if td <= t and not requires_cold[l, td])
        for l in clients for t in T
    }

    frigo_dlv    = {l: 0 for l in clients}
    nonfrigo_dlv = {l: 0 for l in clients}

    x_vars, f_vars, arrival_times, truck_assign, actual_qty, depot_stock, routes_data = {}, {}, {}, {}, {}, {}, {}
    T_last = T[-1]

    priority_deltas = sorted(
        clients, key=lambda l: -abs(priorities.get(l, 0.5) - prev_priorities.get(l, 0.5))
    )
    remaining_moves = k_moves

    for t in T:
        if t == T_last:
            frigo_desired = {l: max(0, frigo_total[l] - frigo_dlv[l]) for l in clients if frigo_total[l] - frigo_dlv[l] > 0}
            nonfrigo_desired = {l: max(0, nonfrigo_total[l] - nonfrigo_dlv[l]) for l in clients if nonfrigo_total[l] - nonfrigo_dlv[l] > 0}
            frigo_floor, nonfrigo_floor = dict(frigo_desired), dict(nonfrigo_desired)
        else:
            frigo_desired, nonfrigo_desired, frigo_floor, nonfrigo_floor = {}, {}, {}, {}
            for l in clients:
                qty = max(0, int(round(quantities[l, t])))
                if requires_cold[l, t]:
                    remaining  = max(0, frigo_total[l] - frigo_dlv[l])
                    min_needed = max(0, cum_frigo_demand[l, t] - frigo_dlv[l])
                    actual     = max(min_needed, min(qty, remaining))
                    if actual > 0 and (min_needed > 0 or actual >= min_delivery):
                        frigo_desired[l] = actual
                        frigo_floor[l]   = min_needed
                else:
                    remaining  = max(0, nonfrigo_total[l] - nonfrigo_dlv[l])
                    min_needed = max(0, cum_nonfrigo_demand[l, t] - nonfrigo_dlv[l])
                    actual     = max(min_needed, min(qty, remaining))
                    if actual > 0 and (min_needed > 0 or actual >= min_delivery):
                        nonfrigo_desired[l] = actual
                        nonfrigo_floor[l]   = min_needed

        max_rel_f  = max(0, int(I_frigo    + R_frigo[t]    - I_min_f))
        max_rel_nf = max(0, int(I_nonfrigo + R_nonfrigo[t] - I_min_nf))
        frigo_cap_total = sum(Q[k] for k in frigo_list)
        nf_cap_total    = sum(Q[k] for k in non_frigo_trucks) if non_frigo_trucks else 0

        min_rel_f  = min(max(0, int(I_frigo    + R_frigo[t]    - I_max_f)), frigo_cap_total)
        min_rel_nf = min(max(0, int(I_nonfrigo + R_nonfrigo[t] - I_max_nf)), nf_cap_total)
        frigo_remaining    = {l: frigo_total[l] - frigo_dlv[l] for l in clients if frigo_total[l] - frigo_dlv[l] > 0}
        nonfrigo_remaining = {l: nonfrigo_total[l] - nonfrigo_dlv[l] for l in clients if nonfrigo_total[l] - nonfrigo_dlv[l] > 0}
        _force_min_release(frigo_desired,    frigo_floor,    frigo_remaining,    min_rel_f)
        _force_min_release(nonfrigo_desired, nonfrigo_floor, nonfrigo_remaining, min_rel_nf)

        frigo_qty    = _clamp_to_integer_budget(frigo_desired,    min(max_rel_f,  frigo_cap_total), frigo_floor)
        nonfrigo_qty = _clamp_to_integer_budget(nonfrigo_desired, min(max_rel_nf, nf_cap_total),    nonfrigo_floor)

        routes_data[t] = {}
        tau_max = params_.get("tau_max", float("inf"))
        prev_routes_t = prev_route_result["routes_data"].get(t, {}) if prev_route_result else {}

        for qty_group, trucks, floor_group in [
            (frigo_qty,    frigo_list,       frigo_floor),
            (nonfrigo_qty, non_frigo_trucks, nonfrigo_floor),
        ]:
            if not qty_group or not trucks:
                continue

            prev_qty_on_route = {}
            prev_truck_of      = {}
            for k in trucks:
                info = prev_routes_t.get(k)
                if not info:
                    continue
                for l_str, q in info["qty"].items():
                    l_int = int(l_str)
                    prev_qty_on_route[l_int] = q
                    prev_truck_of[l_int]     = k

            new_served  = set(qty_group.keys())
            prev_served = set(prev_qty_on_route.keys())

            eligible = prev_route_result is not None and new_served == prev_served
            if eligible:
                load_by_truck = {}
                for l, q in qty_group.items():
                    kk = prev_truck_of[l]
                    load_by_truck[kk] = load_by_truck.get(kk, 0) + q
                if any(load > Q[kk] for kk, load in load_by_truck.items()):
                    eligible = False

            if not eligible:
                stats["fallback"] = stats.get("fallback", 0) + 1
                r, tx, tf, ta, tassign = _nearest_neighbour(
                    qty_group, trucks, t, d, v, s, Q, O, tau_max, floor_group,
                    priorities=priorities,
                )
                routes_data[t].update(r)
                x_vars.update(tx)
                f_vars.update(tf)
                arrival_times.update(ta)
                truck_assign.update(tassign)
                continue

            stats["incremental"] = stats.get("incremental", 0) + 1
            for k in trucks:
                info = prev_routes_t.get(k)
                if not info:
                    continue
                path = list(info["path"])
                qty_on_route = {l: qty_group[l] for l in prev_qty_on_route if prev_truck_of[l] == k}

                if remaining_moves > 0:
                    for l in priority_deltas:
                        if remaining_moves <= 0:
                            break
                        if l not in qty_on_route or prev_truck_of.get(l) != k:
                            continue
                        tau_before = _route_traversal_time(path, k, params_)
                        result = _relocate_client(path, qty_on_route, l, t, k, tau_before, params_)
                        if result is not None:
                            path = result[0]
                        remaining_moves -= 1

                n   = len(path)
                suf = [0] * (n + 1)
                for idx in range(n - 2, -1, -1):
                    node    = path[idx + 1]
                    suf[idx] = suf[idx + 1] + (qty_on_route.get(node, 0) if node != O else 0)

                current_time = 0.0
                current      = O
                for node in path[1:]:
                    current_time += s.get(current, 0.0) + d[current, node] / v[k]
                    if node != O:
                        arrival_times[node, t] = current_time
                    current = node

                for idx in range(n - 1):
                    i, j               = path[idx], path[idx + 1]
                    x_vars[i, j, t, k] = 1
                    f_vars[i, j, t, k] = suf[idx]

                routes_data[t][k] = {"path": path, "qty": {str(l): q for l, q in qty_on_route.items()}}
                truck_assign.update({(l, t): k for l in qty_on_route})

        actually_served_frigo, actually_served_nonfrigo = {}, {}
        for k, info in routes_data[t].items():
            is_frigo = k in frigo_trucks
            for l_str, q in info["qty"].items():
                l_int = int(l_str)
                if is_frigo:
                    actually_served_frigo[l_int]    = actually_served_frigo.get(l_int, 0)    + q
                else:
                    actually_served_nonfrigo[l_int] = actually_served_nonfrigo.get(l_int, 0) + q

        shipped_f = shipped_nf = 0
        for l in clients:
            f, nf = actually_served_frigo.get(l, 0), actually_served_nonfrigo.get(l, 0)
            actual_qty[l, t] = f + nf
            frigo_dlv[l]    += f
            nonfrigo_dlv[l] += nf
            shipped_f += f
            shipped_nf += nf

        I_frigo    = I_frigo    + R_frigo[t]    - shipped_f
        I_nonfrigo = I_nonfrigo + R_nonfrigo[t] - shipped_nf
        depot_stock[t] = {"frigo": round(I_frigo, 4), "nonfrigo": round(I_nonfrigo, 4)}

    tau_return = {}
    for t in T:
        max_ret = 0.0
        for k, info in routes_data.get(t, {}).items():
            path, speed = info["path"], v[k]
            total = sum(s.get(path[idx], 0.0) + d[path[idx], path[idx + 1]] / speed for idx in range(len(path) - 1))
            max_ret = max(max_ret, total)
        tau_return[t] = max_ret

    return {
        "x": x_vars, "f": f_vars, "depot_stock": depot_stock, "arrival_times": arrival_times,
        "truck_assign": truck_assign, "actual_qty": actual_qty, "routes_data": routes_data,
        "tau_return": tau_return,
    }


def _evaluate(x, sets_, params_, prev_route_result, prev_priorities, k_moves, stats):
    quantities, priorities = decode_chromosome(x, sets_)
    if prev_route_result is None:
        route_result = build_routes(quantities, sets_, params_, priorities)
        stats["full_init"] = stats.get("full_init", 0) + 1
    else:
        route_result = _build_routes_incremental(
            quantities, sets_, params_, priorities, prev_route_result, prev_priorities, k_moves, stats,
        )
    F = np.array([
        compute_f1(route_result, sets_, params_), compute_f2(route_result, sets_, params_),
        compute_f3(route_result, sets_, params_), compute_f4(route_result, sets_, params_),
    ])
    G = np.array(_build_g_constraints(route_result, sets_, params_))
    return F, G, route_result, priorities


def run_prototype(instance: str, seed: int, max_gen: int, pop_size: int) -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    sets_, params_ = load_instance(data_path)
    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=N_PARTITIONS)
    effective_pop = max(pop_size, len(ref_dirs))

    alpha_max, alpha_min = 0.10 * np.pi, 0.001 * np.pi
    p_cross, eta_cross = 0.9, 20.0
    eta_mut = 20.0
    migration_period, n_migrate = 10, 10

    problem  = IRPProblem(sets_, params_)
    xl       = np.asarray(problem.xl, dtype=float)
    xu       = np.asarray(problem.xu, dtype=float)
    n_genes  = problem.n_var
    p_mut    = 1.0 / n_genes

    rng      = np.random.default_rng(seed)
    qpop     = QuantumPopulation(effective_pop, n_genes, xl, xu, rng=rng)
    sorter   = NonDominatedSorting()
    survival = ReferenceDirectionSurvival(ref_dirs)
    sbx_op   = SBX(prob=p_cross, eta=eta_cross)
    pm_op    = PM(prob=p_mut,    eta=eta_mut)

    arch_X, arch_F, arch_theta = [], [], []
    _MAX_ARCHIVE = 500

    stats = {}
    infeasible_g_violations = []

    print("=" * 92)
    print(f"  PROTOTYPE DECODEUR INCREMENTAL (sequentiel, pas de campagne) -- {max_gen} gen, seed={seed}")
    print("=" * 92)

    t_start = time.time()

    # ---- generation 0: full decode for everyone (no "previous" route yet) ----
    X_parent = qpop.measure()
    theta_parent = qpop.theta.copy()
    route_pool, priority_pool = [], []
    F_list, G_list = [], []
    for i in range(effective_pop):
        F, G, rr, prio = _evaluate(X_parent[i], sets_, params_, None, None, 0, stats)
        F_list.append(F); G_list.append(G); route_pool.append(rr); priority_pool.append(prio)
    F_parent = np.array(F_list)
    G_parent = np.array(G_list)

    for gen in range(max_gen):
        F_pen_parent = _penalised_F(F_parent, G_parent)
        pareto_idx = sorter.do(F_pen_parent)[0]
        _archive_update(
            X_parent[pareto_idx], F_parent[pareto_idx], G_parent[pareto_idx],
            theta_parent[pareto_idx], arch_X, arch_F, arch_theta, _MAX_ARCHIVE,
        )

        if survival.norm.nadir_point is None:
            F_norm = _normalise_F(F_pen_parent)
        else:
            F_norm = _normalise_F(F_pen_parent, survival.norm.ideal_point, survival.norm.nadir_point)
        assoc = _assign_ref_dirs(F_norm, ref_dirs)
        guides_theta = _select_guides(assoc, pareto_idx, F_norm, ref_dirs, qpop.theta)

        arch_theta_arr = arch_F_norm = None
        if len(arch_X) >= 4:
            arch_theta_arr = np.array(arch_theta)
            if survival.norm.nadir_point is None:
                arch_F_norm = _normalise_F(np.array(arch_F))
            else:
                arch_F_norm = _normalise_F(np.array(arch_F), survival.norm.ideal_point, survival.norm.nadir_point)
            pareto_assoc = assoc[pareto_idx]
            guides_theta = _supplement_from_archive(
                guides_theta, assoc, pareto_assoc, arch_theta_arr, arch_F_norm, ref_dirs,
            )

        alpha = alpha_min + (alpha_max - alpha_min) * (1.0 - gen / max_gen)
        k_moves = max(1, round(alpha / alpha_max * K_MAX))

        qpop.rotate(guides_theta, alpha)
        X_rotated = qpop.measure()

        if p_cross > 0.0:
            idx     = rng.permutation(effective_pop)
            n_pairs = effective_pop // 2
            pairs   = idx[: n_pairs * 2].reshape(n_pairs, 2)
            X_pairs = np.transpose(X_rotated[pairs], (1, 0, 2))
            Qc      = sbx_op._do(problem, X_pairs, random_state=rng)
            X_rotated[pairs[:, 0]] = Qc[0]
            X_rotated[pairs[:, 1]] = Qc[1]
        X_varied = np.clip(pm_op._do(problem, X_rotated, random_state=rng), xl, xu)

        theta_offspring = _encode_theta(X_varied, xl, xu)
        qpop.theta       = theta_offspring
        X_offspring      = qpop.measure()

        off_route_pool, off_priority_pool, F_off_list, G_off_list = [], [], [], []
        for i in range(effective_pop):
            F, G, rr, prio = _evaluate(
                X_offspring[i], sets_, params_, route_pool[i], priority_pool[i], k_moves, stats,
            )
            off_route_pool.append(rr); off_priority_pool.append(prio)
            F_off_list.append(F); G_off_list.append(G)
            if np.any(np.array(G) > 1e-6):
                infeasible_g_violations.append(float(np.max(G)))
        F_offspring = np.array(F_off_list)
        G_offspring = np.array(G_off_list)
        F_pen_offspring = _penalised_F(F_offspring, G_offspring)

        off_pareto_idx = sorter.do(F_pen_offspring)[0]
        _archive_update(
            X_offspring[off_pareto_idx], F_offspring[off_pareto_idx], G_offspring[off_pareto_idx],
            theta_offspring[off_pareto_idx], arch_X, arch_F, arch_theta, _MAX_ARCHIVE,
        )

        theta_pool  = np.vstack([theta_parent, theta_offspring])
        F_true_pool = np.vstack([F_parent, F_offspring])
        G_pool      = np.vstack([G_parent, G_offspring])
        merged_pop  = Population.new(X=theta_pool, F=F_true_pool, G=G_pool)
        merged_pop.set("src_idx", np.arange(2 * effective_pop))
        survived    = survival.do(problem, merged_pop, n_survive=effective_pop, random_state=rng)
        qpop.theta  = np.clip(np.asarray(survived.get("X"), dtype=float), 0.0, np.pi / 2.0)

        src_idx        = np.asarray(survived.get("src_idx"), dtype=int)
        combined_routes = route_pool + off_route_pool
        combined_prios  = priority_pool + off_priority_pool
        combined_F      = np.vstack([F_parent, F_offspring])
        combined_G      = np.vstack([G_parent, G_offspring])
        combined_X      = np.vstack([X_parent, X_offspring])

        # BUGFIX: X_parent used to be re-measured here via qpop.measure(),
        # which draws FRESH random noise (chromosome.py's measure() adds
        # N(0, sigma) noise every call) -- different from whatever X each
        # surviving individual was ACTUALLY evaluated on, desynchronising
        # the (X, F) pairs later passed to _archive_update. Track the exact
        # X used for evaluation instead, reindexed by the same src_idx as
        # everything else -- no re-measurement.
        route_pool    = [combined_routes[j] for j in src_idx]
        priority_pool = [combined_prios[j]  for j in src_idx]
        F_parent      = combined_F[src_idx]
        G_parent      = combined_G[src_idx]
        X_parent       = combined_X[src_idx]
        theta_parent  = qpop.theta.copy()

        if arch_F_norm is not None and migration_period > 0 and gen % migration_period == 0:
            pre_theta = qpop.theta.copy()
            _migrate(qpop, arch_theta_arr, arch_F_norm, assoc, ref_dirs, rng, n_migrate=n_migrate)
            migrated = np.where(np.any(qpop.theta != pre_theta, axis=1))[0]
            for i in migrated:
                Xi = qpop.xl + np.cos(qpop.theta[i]) ** 2 * (qpop.xu - qpop.xl)
                Fi, Gi, rri, prioi = _evaluate(Xi, sets_, params_, None, None, 0, stats)
                route_pool[i], priority_pool[i] = rri, prioi
                F_parent[i], G_parent[i] = Fi, Gi
                X_parent[i] = Xi
            theta_parent = qpop.theta.copy()

        if gen % 5 == 0 or gen == max_gen - 1:
            hv_proxy = np.mean(F_pen_offspring[:, 0])
            print(f"  gen={gen:4d}  alpha={alpha:.4f}  k_moves={k_moves:3d}  "
                  f"incremental={stats.get('incremental',0):5d}  fallback={stats.get('fallback',0):5d}  "
                  f"f1_mean={hv_proxy:9.1f}  archive={len(arch_X)}", flush=True)

    elapsed = time.time() - t_start
    ratio = stats.get('incremental', 0) / max(1, stats.get('incremental', 0) + stats.get('fallback', 0))
    print(f"\n{'-'*92}")
    print(f"  Termine en {elapsed:.1f}s ({max_gen} generations, sequentiel, pop={effective_pop})")
    print(f"  Chemins incrementaux pris : {stats.get('incremental', 0)}  "
          f"Fallback complet : {stats.get('fallback', 0)}  Ratio incrementaux : {ratio:.1%}")
    if infeasible_g_violations:
        print(f"  {len(infeasible_g_violations)} evaluations avec violation de contrainte "
              f"(max={max(infeasible_g_violations):.2f}) -- normal en debut de run, comme en production")
    print(f"{'='*92}\n")

    # ---- final front: same repair_final_front treatment as production, standard decoder ----
    if arch_X:
        pareto_X, _, _ = _crowding_trim(np.array(arch_X), np.array(arch_F), np.array(arch_theta), effective_pop)
    else:
        final_idx = sorter.do(_penalised_F(F_parent, G_parent))[0]
        pareto_X = X_parent[final_idx]

    F_list = []
    for x in pareto_X:
        quantities, priorities = decode_chromosome(x, sets_)
        rr = build_routes(quantities, sets_, params_, priorities)
        rr = _repair_route_result(rr, sets_, params_)
        F_list.append([
            compute_f1(rr, sets_, params_), compute_f2(rr, sets_, params_),
            compute_f3(rr, sets_, params_), compute_f4(rr, sets_, params_),
        ])
    pareto_F = np.array(F_list)

    return pareto_X, pareto_F, elapsed, ratio


def run_comparison(instance: str, seeds, max_gen: int, pop_size: int) -> None:
    from scipy.stats import mannwhitneyu
    from Solvers.NSGA3.metrics import compute_pareto_metrics
    from sensitivity.compare_damped_priority import (
        _load_qinsga3_baseline_from_cache, _load_nsga3_reference, _stats,
    )

    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    sets_, params_ = load_instance(data_path)

    print("=" * 92)
    print(f"  DECODEUR INCREMENTAL (prototype sequentiel) — {len(seeds)} seeds, {max_gen} gen")
    print("=" * 92)

    print("\nDecodage du front NSGA-III (cache, reference fixe)...")
    nsga3_F_runs = _load_nsga3_reference(sets_, params_, instance)
    print(f"  NSGA-III : {len(nsga3_F_runs)} runs, {sum(len(f) for f in nsga3_F_runs)} solutions")

    print("\n>>> baseline (decodeur original, prod. actuelle) -- depuis le cache QINSGA3")
    baseline_runs = _load_qinsga3_baseline_from_cache(sets_, params_, seeds, instance)
    for sd, F, elapsed in baseline_runs:
        print(f"  seed={sd} ... front={len(F)}  time(cache)={elapsed}s")

    print("\n>>> decodeur incremental (test, sequentiel)")
    test_runs = []
    for sd in seeds:
        print(f"  seed={sd} ... ", end="", flush=True)
        pareto_X, pareto_F, elapsed, ratio = run_prototype(instance, sd, max_gen, pop_size)
        test_runs.append((sd, pareto_F, elapsed))
        print(f"front={len(pareto_F)}  time={elapsed:.1f}s  ratio_incremental={ratio:.1%}", flush=True)

    all_F = list(nsga3_F_runs)
    all_F.extend(F for (_, F, _) in baseline_runs if F is not None and len(F) > 0)
    all_F.extend(F for (_, F, _) in test_runs if F is not None and len(F) > 0)
    all_F_stack  = np.vstack(all_F)
    global_ideal = all_F_stack.min(axis=0)
    global_nadir = all_F_stack.max(axis=0)
    print(f"\nIdeal global partage : {global_ideal}")
    print(f"Nadir global partage  : {global_nadir}")

    groups = {}
    nsga3_vals = {"HV": [], "GD": [], "IGD": [], "Spacing": []}
    for F_run in nsga3_F_runs:
        q = compute_pareto_metrics(F_run, global_ideal, global_nadir)
        for k in nsga3_vals:
            nsga3_vals[k].append(q[k])
    groups["NSGA-III (reference)"] = nsga3_vals

    for label, runs in (("baseline (decodeur original)", baseline_runs),
                        ("decodeur incremental (test)", test_runs)):
        vals = {"HV": [], "GD": [], "IGD": [], "Spacing": [], "elapsed_s": []}
        for sd, F, elapsed in runs:
            if F is None or len(F) == 0:
                continue
            q = compute_pareto_metrics(F, global_ideal, global_nadir)
            for k in ("HV", "GD", "IGD", "Spacing"):
                vals[k].append(q[k])
            vals["elapsed_s"].append(elapsed)
        groups[label] = vals

    print(f"\n{'-'*92}\n  RESULTATS\n{'-'*92}")
    for metric, higher in (("HV", True), ("GD", False), ("IGD", False), ("Spacing", False)):
        arrow = "^" if higher else "v"
        print(f"\n  {metric} ({arrow})")
        for name, vals in groups.items():
            s = _stats(vals[metric])
            print(f"    {name:<32} mean={s['mean']:.6f}  std={s['std']:.6f}")

    print(f"\n{'-'*92}\n  Mann-Whitney U : baseline vs decodeur incremental\n{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups["baseline (decodeur original)"][metric], groups["decodeur incremental (test)"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}\n  Temps moyen par run\n{'-'*92}")
    for label in ("baseline (decodeur original)", "decodeur incremental (test)"):
        s = _stats(groups[label]["elapsed_s"])
        print(f"    {label:<32} mean={s['mean']:.1f}s")
    print(f"{'='*92}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prototype sequentiel du decodeur incremental")
    parser.add_argument("--instance", default="100")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--gen",  type=int, default=30)
    parser.add_argument("--pop",  type=int, default=200)
    args = parser.parse_args()

    if len(args.seeds) == 1:
        run_prototype(args.instance, args.seeds[0], args.gen, args.pop)
    else:
        run_comparison(args.instance, args.seeds, args.gen, args.pop)
