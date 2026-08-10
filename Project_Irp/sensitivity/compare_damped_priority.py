"""Damped-priority decoder test for QINSGA-III.

Context: `sensitivity/diagnose_giant_tour_continuity.py` tried replacing the
greedy nearest-neighbour decoder's construction principle entirely (giant
tour + capacity split) to restore theta->route continuity -- failed on both
axes (chaos WORSE at every alpha tested, ~4x worse base f1), because the
split step turned out to be just as sequential/cascading as the original,
just relocated from distance to capacity-boundary.

This tests a smaller, surgical change instead: KEEP the exact greedy,
distance-driven construction (so route quality stays close to production),
but DAMP how much a priority perturbation can move the decoder's score
(`Solvers/NSGA3/decoder.py:331`, `score = dist / (0.5 + priority)`) by
interpolating priority toward a neutral constant:
    score = dist / (0.5 + w*priority + (1-w)*0.5),  w in [0, 1]
w=1.0 is byte-identical to production. Smaller w means priority can only
break already-close distance ties, never flip a clearly-better geographic
choice -- damping the mechanism that lets a small theta perturbation cascade
into a disproportionately different route.

Smoke test (`diagnose_damped_priority_continuity.py`, real cached NSGA-III
100-client baselines, 5 seeds): w=0.2 cut Jaccard route-distance ~14% at
large/mid rotation-schedule alpha (start/middle of a run) for a modest +8%
f1 cost on unperturbed baselines -- no effect at the smallest alpha (end of
run), unlike giant-tour's uniform failure. First candidate showing a real,
non-catastrophic trade-off; this script promotes it to the standard
protocol (3 seeds, shared ideal/nadir, Mann-Whitney U) used by every other
remedy in this project.

Isolation: `Solvers/NSGA3/decoder.py` is NEVER modified -- full local forks
of `build_routes`/`IRPProblem`/`run_qinsga3`, wired into a private
evaluation path used ONLY by this script's QI-NSGA-III test variant.
NSGA-III's reference numbers come from the existing chromosome cache,
decoded with the UNMODIFIED shared decoder -- unaffected. Same isolation
pattern as `compare_lookahead_decoder.py`.

Usage:
    python -m sensitivity.compare_damped_priority --gen 10 --seeds 42   # cost smoke test
    python -m sensitivity.compare_damped_priority --seeds 42 137 271 --gen 300 --w 0.2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODULE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.population import Population
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from scipy.stats import mannwhitneyu

from models.parametres          import load_instance
from Solvers.NSGA3.decoder       import (
    decode_chromosome, build_routes,
    _force_min_release, _clamp_to_integer_budget,
)
from Solvers.NSGA3.evaluator     import compute_f1, compute_f2, compute_f3, compute_f4
from Solvers.NSGA3.metrics       import compute_pareto_metrics
from Solvers.NSGA3.problem       import IRPProblem
from Solvers.QINSGA3.algorithm   import (
    _encode_theta, _penalised_F,
    _normalise_F, _assign_ref_dirs, _select_guides, _supplement_from_archive,
    _migrate, _crowding_trim, _archive_update, _build_g_constraints,
)
from Solvers.QINSGA3.chromosome  import QuantumPopulation
from Solvers.QINSGA3.repair      import _repair_route_result

N_PARTITIONS = 8
N_OBJ        = 4
POP_SIZE     = 200

DEFAULT_SEEDS = [42, 137, 271]
DEFAULT_W     = 0.2

_NSGA3_CACHE   = os.path.join(PROJECT_DIR, "Solvers", "NSGA3",   "nsga3_chromosomes.json")
_QINSGA3_CACHE = os.path.join(PROJECT_DIR, "Solvers", "QINSGA3", "qinsga3_chromosomes.json")


# ---------------------------------------------------------------------------
# Damped-priority decoder (local fork of Solvers/NSGA3/decoder.py's construction step)
# ---------------------------------------------------------------------------

def _nearest_neighbour_damped(qty_dict, trucks, t, d, v, s, Q, O, tau_max, floors, priorities, w):
    """Same as decoder.py::_nearest_neighbour, except non-mandatory candidate
    scores use priority damped toward the neutral constant 0.5 by factor w.
    Mandatory (floor>0) candidates keep the original undamped score -- the
    floor commitment must still be honoured on schedule, damping it would
    risk delivery deadlines for no diversity benefit (mandatory clients
    aren't where the rotation-driven priority exploration matters)."""
    x_vars, f_vars, arrivals, assign, routes = {}, {}, {}, {}, {}
    prios     = priorities or {}
    pending   = dict(qty_dict)
    truck_idx = 0

    while pending and truck_idx < len(trucks):
        k, cap, speed = trucks[truck_idx], Q[trucks[truck_idx]], v[trucks[truck_idx]]
        path, qty_on_route, load, current_time, current = [O], {}, 0, 0.0, O

        while True:
            best, best_score, best_qty = None, float("inf"), 0
            has_mandatory = False
            any_mandatory_pending = any((floors or {}).get(l2, 0) > 0 for l2 in pending)

            for l, q in pending.items():
                floor_l      = (floors or {}).get(l, 0)
                is_mandatory = floor_l > 0
                if not is_mandatory and any_mandatory_pending:
                    continue

                if load + q <= cap:
                    q_effective = q
                elif is_mandatory and load + floor_l <= cap:
                    q_effective = floor_l
                else:
                    continue

                dist = d[current, l]
                if tau_max is not None and not is_mandatory:
                    projected = (current_time + s.get(current, 0.0) + dist / speed
                                 + s.get(l, 0.0) + d[l, O] / speed)
                    if projected > tau_max:
                        continue

                p_raw = prios.get(l, 0.5)
                p_eff = p_raw if is_mandatory else (w * p_raw + (1 - w) * 0.5)
                score = dist / (0.5 + p_eff)

                if is_mandatory:
                    if not has_mandatory or score < best_score:
                        best_score, best, best_qty = score, l, q_effective
                        has_mandatory = True
                elif not has_mandatory and score < best_score:
                    best_score, best, best_qty = score, l, q_effective

            if best is None:
                break
            current_time      += s.get(current, 0.0) + d[current, best] / speed
            path.append(best)
            pending.pop(best)
            qty_on_route[best] = best_qty
            load              += qty_on_route[best]
            arrivals[best, t]  = current_time
            assign[best, t]    = k
            current            = best

        path.append(O)
        if len(path) > 2:
            n   = len(path)
            suf = [0] * (n + 1)
            for idx in range(n - 2, -1, -1):
                node    = path[idx + 1]
                suf[idx] = suf[idx + 1] + (qty_on_route.get(node, 0) if node != O else 0)
            for idx in range(n - 1):
                i, j               = path[idx], path[idx + 1]
                x_vars[i, j, t, k] = 1
                f_vars[i, j, t, k] = suf[idx]
            routes[k] = {"path": path, "qty": {str(l): q for l, q in qty_on_route.items()}}
        truck_idx += 1

    return routes, x_vars, f_vars, arrivals, assign


def build_routes_damped(quantities, sets_, params_, priorities, w):
    """Local fork of Solvers/NSGA3/decoder.py::build_routes, calling
    _nearest_neighbour_damped instead of the original _nearest_neighbour.
    Everything else identical -- imported and reused unchanged."""
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
        for qty_group, trucks, floor_group in [
            (frigo_qty,    frigo_list,       frigo_floor),
            (nonfrigo_qty, non_frigo_trucks, nonfrigo_floor),
        ]:
            if not qty_group or not trucks:
                continue
            r, tx, tf, ta, tassign = _nearest_neighbour_damped(
                qty_group, trucks, t, d, v, s, Q, O, tau_max, floor_group, priorities, w,
            )
            routes_data[t].update(r)
            x_vars.update(tx)
            f_vars.update(tf)
            arrival_times.update(ta)
            truck_assign.update(tassign)

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


class IRPProblemDamped(ElementwiseProblem):
    """Local fork of Solvers/NSGA3/problem.py::IRPProblem, using
    build_routes_damped instead of the shared decoder's build_routes."""

    def __init__(self, sets_, params_, w):
        clients = sets_["clients"]
        T       = sets_["T"]
        q_lt    = params_["q_lt"]
        T_first = T[0]

        xl, xu = [], []
        for l in clients:
            total_l = float(sum(q_lt[l, t] for t in T))
            for t in T:
                xl.append(float(q_lt[l, T_first]) if t == T_first else 0.0)
                xu.append(total_l)

        n_prio = len(clients)
        xl += [0.0] * n_prio
        xu += [1.0] * n_prio

        super().__init__(
            n_var=len(clients) * len(T) + n_prio, n_obj=4,
            n_ieq_constr=2 * len(T) + len(clients) * len(T) + 2 * len(T),
            xl=xl, xu=xu,
        )
        self.sets_, self.params_, self.w = sets_, params_, w

    def _evaluate(self, x, out, *args, **kwargs):
        quantities, priorities = decode_chromosome(x, self.sets_)
        route_result = build_routes_damped(quantities, self.sets_, self.params_, priorities, self.w)

        out["F"] = [
            compute_f1(route_result, self.sets_, self.params_), compute_f2(route_result, self.sets_, self.params_),
            compute_f3(route_result, self.sets_, self.params_), compute_f4(route_result, self.sets_, self.params_),
        ]

        clients, T, q_lt = self.sets_["clients"], self.sets_["T"], self.params_["q_lt"]
        tau_min, tau_max = self.params_["tau_min"], self.params_["tau_max"]
        I_max_f, I_max_nf = self.params_["I_O_max_frigo"], self.params_["I_O_max_nonfrigo"]
        actual, depot_stock = route_result["actual_qty"], route_result["depot_stock"]

        G = []
        for t in T:
            ret = route_result["tau_return"].get(t, 0.0)
            G.append(ret - tau_max)
            G.append(tau_min - ret)
        for l in clients:
            cum_del = cum_dem = 0
            for t in T:
                cum_del += actual.get((l, t), 0)
                cum_dem += q_lt[l, t]
                G.append(cum_dem - cum_del)
        for t in T:
            G.append(depot_stock[t]["frigo"]    - I_max_f)
            G.append(depot_stock[t]["nonfrigo"] - I_max_nf)

        out["G"] = G


_g_problem = None


def _worker_init_damped(sets_, params_, w):
    global _g_problem
    _g_problem = IRPProblemDamped(sets_, params_, w)


def _worker_eval_damped(x):
    out = {}
    _g_problem._evaluate(x, out)
    return out["F"], out["G"]


def _repair_pareto_front_damped(pareto_X, sets_, params_, w):
    """Damped-decoder equivalent of Solvers/QINSGA3/algorithm.py's
    _repair_pareto_front (repair_final_front=True path) -- decodes each
    final-front chromosome with build_routes_damped (not the standard
    decoder, to stay consistent with what the search itself just optimised
    against), then applies the SAME production repair (_repair_route_result,
    decoder-agnostic -- it 2-opts an already-built route_result by real f1
    cost, never re-invokes the decoder's score formula). Without this step,
    the damped-priority arm would be missing the one correction already
    proven to add ~+39% HV in this project, making any A/B comparison
    against the (repaired) production baseline invalid."""
    F_list, G_list = [], []
    for x in pareto_X:
        quantities, priorities = decode_chromosome(x, sets_)
        route_result = build_routes_damped(quantities, sets_, params_, priorities, w)
        route_result = _repair_route_result(route_result, sets_, params_)
        F = np.array([
            compute_f1(route_result, sets_, params_), compute_f2(route_result, sets_, params_),
            compute_f3(route_result, sets_, params_), compute_f4(route_result, sets_, params_),
        ])
        G = np.array(_build_g_constraints(route_result, sets_, params_))
        F_list.append(F)
        G_list.append(G)
    return np.array(F_list), np.array(G_list)


def run_qinsga3_damped_priority(
    sets_, params_, ref_dirs,
    pop_size=200, max_gen=300,
    alpha_max=0.10 * np.pi, alpha_min=0.001 * np.pi,
    p_cross=0.9, eta_cross=20.0,
    p_mut=None, eta_mut=20.0,
    migration_period=10, n_migrate=10,
    w=DEFAULT_W,
    repair_final_front=True,
    seed=42, rotation_type="tanh", callback=None,
):
    """Identical to production Solvers/QINSGA3/algorithm.py::run_qinsga3,
    except IRPProblem is replaced by IRPProblemDamped (priority-damped
    decoder score) for the search itself. repair_final_front=True (matching
    production's default) applies _repair_pareto_front_damped once on the
    returned front -- see its docstring for why this must not be skipped."""
    rng     = np.random.default_rng(seed)
    problem = IRPProblem(sets_, params_)   # metadata only (xl/xu/SBX/PM bounds)

    xl       = np.asarray(problem.xl, dtype=float)
    xu       = np.asarray(problem.xu, dtype=float)
    n_genes  = problem.n_var
    n_constr = problem.n_ieq_constr

    if p_mut is None:
        p_mut = 1.0 / n_genes

    qpop     = QuantumPopulation(pop_size, n_genes, xl, xu, rng=rng, rotation_type=rotation_type)
    sorter   = NonDominatedSorting()
    survival = ReferenceDirectionSurvival(ref_dirs)
    sbx_op   = SBX(prob=p_cross, eta=eta_cross)
    pm_op    = PM(prob=p_mut,    eta=eta_mut)

    arch_X, arch_F, arch_theta = [], [], []
    _MAX_ARCHIVE = 500

    n_workers = min(os.cpu_count() or 1, pop_size)
    chunksize = max(1, pop_size // (2 * n_workers))

    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_worker_init_damped,
        initargs=(sets_, params_, w),
    ) as pool:

        def _eval_batch(X):
            results = list(pool.map(_worker_eval_damped, list(X), chunksize=chunksize))
            return (np.array([r[0] for r in results]), np.array([r[1] for r in results]))

        for gen in range(max_gen):
            theta_parent = qpop.theta.copy()
            X_parent     = qpop.measure()
            F_parent, G_parent = _eval_batch(X_parent)
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
            assoc  = _assign_ref_dirs(F_norm, ref_dirs)
            guides_theta = _select_guides(assoc, pareto_idx, F_norm, ref_dirs, qpop.theta)

            arch_theta_arr = None
            arch_F_norm    = None
            if len(arch_X) >= 4:
                arch_theta_arr = np.array(arch_theta)
                if survival.norm.nadir_point is None:
                    arch_F_norm = _normalise_F(np.array(arch_F))
                else:
                    arch_F_norm = _normalise_F(np.array(arch_F), survival.norm.ideal_point, survival.norm.nadir_point)
                pareto_assoc   = assoc[pareto_idx]
                guides_theta   = _supplement_from_archive(
                    guides_theta, assoc, pareto_assoc, arch_theta_arr, arch_F_norm, ref_dirs,
                )

            alpha = alpha_min + (alpha_max - alpha_min) * (1.0 - gen / max_gen)
            qpop.rotate(guides_theta, alpha)
            X_rotated = qpop.measure()

            if p_cross > 0.0:
                idx     = rng.permutation(pop_size)
                n_pairs = pop_size // 2
                pairs   = idx[: n_pairs * 2].reshape(n_pairs, 2)
                X_pairs = np.transpose(X_rotated[pairs], (1, 0, 2))
                Qc      = sbx_op._do(problem, X_pairs, random_state=rng)
                X_rotated[pairs[:, 0]] = Qc[0]
                X_rotated[pairs[:, 1]] = Qc[1]
            X_varied = np.clip(pm_op._do(problem, X_rotated, random_state=rng), xl, xu)

            theta_offspring = _encode_theta(X_varied, xl, xu)
            qpop.theta      = theta_offspring
            X_offspring     = qpop.measure()
            F_offspring, G_offspring = _eval_batch(X_offspring)
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
            survived    = survival.do(problem, merged_pop, n_survive=pop_size, random_state=rng)
            qpop.theta  = np.clip(np.asarray(survived.get("X"), dtype=float), 0.0, np.pi / 2.0)

            if (arch_F_norm is not None and migration_period > 0 and gen % migration_period == 0):
                _migrate(qpop, arch_theta_arr, arch_F_norm, assoc, ref_dirs, rng, n_migrate=n_migrate)

            if callback is not None and (gen % 10 == 0 or gen == max_gen - 1):
                callback(gen, F_offspring, G_offspring, off_pareto_idx)

        X_final = qpop.measure()
        F_final, G_final = _eval_batch(X_final)
        F_pen_final      = _penalised_F(F_final, G_final)
        final_pareto_idx = sorter.do(F_pen_final)[0]
        _archive_update(
            X_final[final_pareto_idx], F_final[final_pareto_idx], G_final[final_pareto_idx],
            qpop.theta[final_pareto_idx], arch_X, arch_F, arch_theta, _MAX_ARCHIVE,
        )

    if arch_X:
        pareto_X, pareto_F, pareto_G = _crowding_trim(
            np.array(arch_X), np.array(arch_F), np.array(arch_theta), pop_size,
        )
        pareto_G = np.zeros((len(pareto_X), n_constr))
    else:
        pareto_X, pareto_F, pareto_G = (
            X_final[final_pareto_idx], F_final[final_pareto_idx], G_final[final_pareto_idx])

    if repair_final_front:
        pareto_F, pareto_G = _repair_pareto_front_damped(pareto_X, sets_, params_, w)

    return pareto_X, pareto_F, pareto_G


# ---------------------------------------------------------------------------
# Comparison harness
# ---------------------------------------------------------------------------

def _decode_to_F(chromosomes, sets_, params_) -> np.ndarray:
    F = []
    for chrom in chromosomes:
        quantities, priorities = decode_chromosome(np.array(chrom), sets_)
        route_result = build_routes(quantities, sets_, params_, priorities)
        F.append([
            compute_f1(route_result, sets_, params_), compute_f2(route_result, sets_, params_),
            compute_f3(route_result, sets_, params_), compute_f4(route_result, sets_, params_),
        ])
    return np.array(F)


def _load_nsga3_reference(sets_, params_, instance: str):
    with open(_NSGA3_CACHE, encoding="utf-8") as f:
        cache = json.load(f)
    expected = f"{instance}_clients"
    cached   = cache["runs"][0]["meta_base"]["instance"]
    if cached != expected:
        raise ValueError(f"NSGA-III cache is for '{cached}', not '{expected}'")
    return [_decode_to_F(r["chromosomes"], sets_, params_) for r in cache["runs"]]


def _decode_repaired_to_F(chromosomes, sets_, params_) -> np.ndarray:
    """Same as _decode_to_F, but also applies _repair_route_result -- matches
    current production (repair_final_front=True, best-improvement, the
    default `run_qinsga3` behaviour since 2026-08-08). Needed to reproduce
    what a fresh `run_qinsga3` baseline run would report, from chromosomes
    already cached (Baldwinian repair is fitness-only, never re-encoded into
    X -- decoding without also repairing would silently reproduce the
    PRE-remedy-G front instead)."""
    F = []
    for chrom in chromosomes:
        quantities, priorities = decode_chromosome(np.array(chrom), sets_)
        route_result = build_routes(quantities, sets_, params_, priorities)
        route_result = _repair_route_result(route_result, sets_, params_)
        F.append([
            compute_f1(route_result, sets_, params_), compute_f2(route_result, sets_, params_),
            compute_f3(route_result, sets_, params_), compute_f4(route_result, sets_, params_),
        ])
    return np.array(F)


def _load_qinsga3_baseline_from_cache(sets_, params_, seeds, instance: str):
    """Reuses Solvers/QINSGA3/qinsga3_chromosomes.json instead of re-running
    a fresh 300-gen QI-NSGA-III search for the baseline arm -- the cache
    already holds the exact seeds this project's other comparisons use
    (42, 137, 271, ... 20 seeds total), decoded+repaired here to match
    current production exactly."""
    with open(_QINSGA3_CACHE, encoding="utf-8") as f:
        cache = json.load(f)
    by_seed  = {r["seed"]: r for r in cache["runs"]}
    expected = f"{instance}_clients"

    missing = [sd for sd in seeds if sd not in by_seed]
    if missing:
        raise ValueError(f"seeds {missing} not present in QINSGA3 cache "
                          f"(available: {sorted(by_seed.keys())})")

    out = []
    for sd in seeds:
        run = by_seed[sd]
        cached_instance = run["meta_base"]["instance"]
        if cached_instance != expected:
            raise ValueError(f"QINSGA3 cache seed {sd} is for '{cached_instance}', not '{expected}'")
        F = _decode_repaired_to_F(run["chromosomes"], sets_, params_)
        elapsed = run["meta_base"].get("elapsed_s")
        out.append((sd, F, elapsed))
    return out


def _stats(vals):
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def run_comparison(instance: str, seeds, max_gen: int, pop_size: int, w: float) -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    sets_, params_ = load_instance(data_path)
    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=N_PARTITIONS)
    effective_pop = max(pop_size, len(ref_dirs))

    print("=" * 92)
    print(f"  PRIORITE AMORTIE (w={w}) — QINSGA-III (baseline vs NSGA-III cache)")
    print("=" * 92)
    print(f"  Instance : {instance} clients | pop={effective_pop} | gen={max_gen}")
    print(f"  Seeds    : {seeds}")
    print("=" * 92)

    print("\nDecodage du front NSGA-III (cache, reference fixe, decodeur ORIGINAL)...")
    nsga3_F_runs = _load_nsga3_reference(sets_, params_, instance)
    print(f"  NSGA-III : {len(nsga3_F_runs)} runs, {sum(len(f) for f in nsga3_F_runs)} solutions")

    raw = {"baseline (decodeur original)": [], f"priorite amortie w={w} (test)": []}

    print("\n>>> baseline (decodeur original, prod. actuelle) -- depuis le cache QINSGA3, pas de re-run")
    for seed, F, elapsed in _load_qinsga3_baseline_from_cache(sets_, params_, seeds, instance):
        raw["baseline (decodeur original)"].append((seed, F, elapsed))
        print(f"  seed={seed} ... front={len(F)}  time(cache)={elapsed}s", flush=True)

    print(f"\n>>> priorite amortie w={w} (test)")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        X, pareto_F, _ = run_qinsga3_damped_priority(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed, w=w,
        )
        elapsed = round(time.time() - t0, 1)
        raw[f"priorite amortie w={w} (test)"].append((seed, pareto_F, elapsed))
        print(f"front={len(pareto_F) if pareto_F is not None else 0}  time={elapsed}s", flush=True)

    all_F = list(nsga3_F_runs)
    for lbl in raw:
        all_F.extend(F for (_, F, _) in raw[lbl] if F is not None and len(F) > 0)
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

    for lbl in ("baseline (decodeur original)", f"priorite amortie w={w} (test)"):
        vals = {"HV": [], "GD": [], "IGD": [], "Spacing": [], "elapsed_s": [], "front_size": []}
        for seed, F, elapsed in raw[lbl]:
            if F is None or len(F) == 0:
                continue
            q = compute_pareto_metrics(F, global_ideal, global_nadir)
            for k in ("HV", "GD", "IGD", "Spacing"):
                vals[k].append(q[k])
            vals["elapsed_s"].append(elapsed)
            vals["front_size"].append(len(F))
        groups[lbl] = vals

    print(f"\n{'-'*92}")
    print("  RESULTATS")
    print(f"{'-'*92}")
    for metric, higher in (("HV", True), ("GD", False), ("IGD", False), ("Spacing", False)):
        arrow = "^" if higher else "v"
        print(f"\n  {metric} ({arrow})")
        for name, vals in groups.items():
            s = _stats(vals[metric])
            print(f"    {name:<32} mean={s['mean']:.6f}  std={s['std']:.6f}")

    print(f"\n{'-'*92}")
    print(f"  Mann-Whitney U : baseline vs priorite amortie w={w}")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups["baseline (decodeur original)"][metric], groups[f"priorite amortie w={w} (test)"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print(f"  Mann-Whitney U : priorite amortie w={w} vs NSGA-III")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups[f"priorite amortie w={w} (test)"][metric], groups["NSGA-III (reference)"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print("  Temps moyen par run  (SURCOUT DU DECODEUR A SURVEILLER)")
    print(f"{'-'*92}")
    for lbl in ("baseline (decodeur original)", f"priorite amortie w={w} (test)"):
        s = _stats(groups[lbl]["elapsed_s"])
        print(f"    {lbl:<32} mean={s['mean']:.1f}s")

    print(f"\n{'='*92}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de priorite amortie (score decodeur) — QINSGA-III")
    parser.add_argument("--instance", default="100", choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen",   type=int, default=300)
    parser.add_argument("--pop",   type=int, default=POP_SIZE)
    parser.add_argument("--w",     type=float, default=DEFAULT_W)
    args = parser.parse_args()

    run_comparison(args.instance, args.seeds, args.gen, args.pop, args.w)
