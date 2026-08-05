"""Softened-decoder test for QINSGA-III (experiment A, "chaotic decoder" fix).

Context: a local-sensitivity test (sensitivity/test_theta_route_sensitivity.py)
found that Solvers/NSGA3/decoder.py's greedy nearest-neighbour construction
is highly discontinuous: a small theta perturbation that DOES change the
decoded route often produces a huge, disproportionate jump in objective
space (median |Delta F| in the hundreds to low thousands, vs f1 in the
15000-18000 range). Root cause: _nearest_neighbour() picks the strict
argmin(score) candidate at every step -- a small score change can flip
which client is picked, and since the construction never backtracks, that
one flip cascades through the rest of the route (classic greedy-heuristic
butterfly effect).

Fix tested here: replace the strict argmin with a SOFTENED (Boltzmann/
softmax) selection among non-mandatory candidates -- P(l) ~ exp(-(score_l -
min_score) / T), T = SOFTEN_TEMP_FRAC * min_score. Small score differences
now shift probabilities smoothly instead of flipping a hard decision
boundary; large score differences still behave almost like the original
argmin (very low probability for clearly-worse candidates). Mandatory
clients (floor > 0, delivery-deadline-bound) keep the ORIGINAL deterministic
argmin -- softening a hard delivery deadline is not the same risk/reward as
softening a routing preference.

Isolation from NSGA-III: this is a full local fork of build_routes/
_nearest_neighbour (not a monkeypatch of the shared Solvers/NSGA3/decoder.py
module) wired into a private IRPProblem-like evaluation path used ONLY by
this script's QI-NSGA-III test variant. NSGA-III's reference numbers come
from the existing chromosome cache, decoded with the UNMODIFIED shared
decoder (Solvers.NSGA3.decoder), exactly like every other comparison script
this session -- NSGA-III's own results are completely unaffected.

Usage:
    python -m sensitivity.compare_soft_decoder
    python -m sensitivity.compare_soft_decoder --seeds 42 137 271 --gen 300 --temp 0.15
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
    run_qinsga3, _encode_theta, _penalised_F,
    _normalise_F, _assign_ref_dirs, _select_guides, _supplement_from_archive,
    _migrate, _crowding_trim, _archive_update,
)
from Solvers.QINSGA3.chromosome  import QuantumPopulation

N_PARTITIONS = 8
N_OBJ        = 4
POP_SIZE     = 200

DEFAULT_SEEDS = [42, 137, 271]
DEFAULT_TEMP  = 0.15  # softening temperature, as a fraction of the best score

_NSGA3_CACHE = os.path.join(PROJECT_DIR, "Solvers", "NSGA3", "nsga3_chromosomes.json")


# ---------------------------------------------------------------------------
# Softened decoder (local fork of Solvers/NSGA3/decoder.py's construction step)
# ---------------------------------------------------------------------------

def _nearest_neighbour_soft(qty_dict, trucks, t, d, v, s, Q, O, tau_max, floors, priorities, rng, temp_frac):
    """Same as Solvers/NSGA3/decoder.py::_nearest_neighbour, except the
    non-mandatory candidate is chosen via softened (Boltzmann) sampling over
    all feasible-at-this-step candidates instead of strict argmin(score).
    Mandatory (floor>0) candidates keep the original deterministic argmin --
    softening a hard delivery deadline is not the same risk as softening a
    routing preference."""
    x_vars, f_vars, arrivals, assign, routes = {}, {}, {}, {}, {}
    prios     = priorities or {}
    pending   = dict(qty_dict)
    truck_idx = 0

    while pending and truck_idx < len(trucks):
        k, cap, speed = trucks[truck_idx], Q[trucks[truck_idx]], v[trucks[truck_idx]]
        path, qty_on_route, load, current_time, current = [O], {}, 0, 0.0, O

        while True:
            any_mandatory_pending = any((floors or {}).get(l2, 0) > 0 for l2 in pending)

            mand_best, mand_score, mand_qty = None, float("inf"), 0
            soft_candidates = []  # (l, score, q_effective)

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

                score = dist / (0.5 + prios.get(l, 0.5))
                if is_mandatory:
                    if score < mand_score:
                        mand_score, mand_best, mand_qty = score, l, q_effective
                else:
                    soft_candidates.append((l, score, q_effective))

            if mand_best is not None:
                best, best_qty = mand_best, mand_qty
            elif soft_candidates:
                scores = np.array([c[1] for c in soft_candidates])
                min_s  = scores.min()
                temp   = max(temp_frac * max(min_s, 1e-6), 1e-6)
                logits = -(scores - min_s) / temp
                probs  = np.exp(logits - logits.max())
                probs /= probs.sum()
                pick = rng.choice(len(soft_candidates), p=probs)
                best, _, best_qty = soft_candidates[pick]
            else:
                break

            current_time      += s.get(current, 0.0) + d[current, best] / speed
            path.append(best)
            pending.pop(best)
            qty_on_route[best] = best_qty
            load               += qty_on_route[best]
            arrivals[best, t]   = current_time
            assign[best, t]     = k
            current              = best

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


def build_routes_soft(quantities, sets_, params_, priorities, rng, temp_frac):
    """Local fork of Solvers/NSGA3/decoder.py::build_routes, calling
    _nearest_neighbour_soft instead of the original _nearest_neighbour.
    Everything else (demand bookkeeping, stock ceilings, floors) is
    identical -- imported and reused unchanged from the shared decoder."""
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
            r, tx, tf, ta, tassign = _nearest_neighbour_soft(
                qty_group, trucks, t, d, v, s, Q, O, tau_max, floor_group, priorities, rng, temp_frac,
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


class IRPProblemSoft(ElementwiseProblem):
    """Local fork of Solvers/NSGA3/problem.py::IRPProblem, using
    build_routes_soft instead of the shared decoder's build_routes. Identical
    objectives/constraints logic otherwise."""

    def __init__(self, sets_, params_, rng, temp_frac):
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
        self.sets_, self.params_, self.rng, self.temp_frac = sets_, params_, rng, temp_frac

    def _evaluate(self, x, out, *args, **kwargs):
        quantities, priorities = decode_chromosome(x, self.sets_)
        route_result = build_routes_soft(quantities, self.sets_, self.params_, priorities, self.rng, self.temp_frac)

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


def _worker_init_soft(sets_, params_, worker_seed, temp_frac):
    global _g_problem
    _g_problem = IRPProblemSoft(sets_, params_, np.random.default_rng(worker_seed), temp_frac)


def _worker_eval_soft(x):
    out = {}
    _g_problem._evaluate(x, out)
    return out["F"], out["G"]


def run_qinsga3_soft_decoder(
    sets_, params_, ref_dirs,
    pop_size=200, max_gen=300,
    alpha_max=0.10 * np.pi, alpha_min=0.001 * np.pi,
    p_cross=0.9, eta_cross=20.0,
    p_mut=None, eta_mut=20.0,
    migration_period=10, n_migrate=10,
    temp_frac=DEFAULT_TEMP,
    seed=42, rotation_type="tanh", callback=None,
):
    """Identical to production Solvers/QINSGA3/algorithm.py::run_qinsga3,
    except IRPProblem is replaced by IRPProblemSoft (softened decoder
    selection) for the search itself. Uses IRPProblem (the real, unmodified
    one) only for xl/xu/n_var/n_ieq_constr metadata in the main process,
    matching production exactly."""
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
        max_workers=n_workers, initializer=_worker_init_soft,
        initargs=(sets_, params_, seed + 999, temp_frac),
    ) as pool:

        def _eval_batch(X):
            results = list(pool.map(_worker_eval_soft, list(X), chunksize=chunksize))
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
                Q       = sbx_op._do(problem, X_pairs, random_state=rng)
                X_rotated[pairs[:, 0]] = Q[0]
                X_rotated[pairs[:, 1]] = Q[1]
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
        arch_X_arr, arch_F_arr, _ = _crowding_trim(
            np.array(arch_X), np.array(arch_F), np.array(arch_theta), pop_size,
        )
        return arch_X_arr, arch_F_arr, np.zeros((len(arch_X_arr), n_constr))
    return X_final[final_pareto_idx], F_final[final_pareto_idx], G_final[final_pareto_idx]


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


def _stats(vals):
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def _chrom_diversity(X_list, xl, xu):
    X = np.array(X_list, dtype=float)
    denom = np.where(xu - xl > 1e-9, xu - xl, 1.0)
    P = (X - xl) / denom
    return float(P.var(axis=0).mean())


def run_comparison(instance: str, seeds, max_gen: int, pop_size: int, temp_frac: float) -> None:
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{instance}_clients.json")
    sets_, params_ = load_instance(data_path)
    problem_ref = IRPProblem(sets_, params_)
    xl = np.asarray(problem_ref.xl, dtype=float)
    xu = np.asarray(problem_ref.xu, dtype=float)
    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=N_PARTITIONS)
    effective_pop = max(pop_size, len(ref_dirs))

    print("=" * 92)
    print(f"  DECODEUR ADOUCI (temp_frac={temp_frac}) — QINSGA-III (baseline vs NSGA-III cache)")
    print("=" * 92)
    print(f"  Instance : {instance} clients | pop={effective_pop} | gen={max_gen}")
    print(f"  Seeds    : {seeds}")
    print("=" * 92)

    print("\nDecodage du front NSGA-III (cache, reference fixe, decodeur ORIGINAL)...")
    nsga3_F_runs = _load_nsga3_reference(sets_, params_, instance)
    print(f"  NSGA-III : {len(nsga3_F_runs)} runs, {sum(len(f) for f in nsga3_F_runs)} solutions")

    raw       = {"baseline (decodeur original)": [], "decodeur adouci (test)": []}
    chrom_X   = {"baseline (decodeur original)": [], "decodeur adouci (test)": []}

    print("\n>>> baseline (decodeur original, prod. actuelle)")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        X, pareto_F, _ = run_qinsga3(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed,
        )
        elapsed = round(time.time() - t0, 1)
        raw["baseline (decodeur original)"].append((seed, pareto_F, elapsed))
        chrom_X["baseline (decodeur original)"].extend(X.tolist())
        print(f"front={len(pareto_F) if pareto_F is not None else 0}  time={elapsed}s", flush=True)

    print("\n>>> decodeur adouci (test)")
    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        t0 = time.time()
        X, pareto_F, _ = run_qinsga3_soft_decoder(
            sets_=sets_, params_=params_, ref_dirs=ref_dirs,
            pop_size=effective_pop, max_gen=max_gen, seed=seed, temp_frac=temp_frac,
        )
        elapsed = round(time.time() - t0, 1)
        raw["decodeur adouci (test)"].append((seed, pareto_F, elapsed))
        chrom_X["decodeur adouci (test)"].extend(X.tolist())
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

    for lbl in ("baseline (decodeur original)", "decodeur adouci (test)"):
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
            print(f"    {name:<30} mean={s['mean']:.6f}  std={s['std']:.6f}")

    print(f"\n{'-'*92}")
    print("  Diversite chromosome  Var(X_norm)  (pooled)")
    print(f"{'-'*92}")
    for lbl in ("baseline (decodeur original)", "decodeur adouci (test)"):
        d = _chrom_diversity(chrom_X[lbl], xl, xu)
        print(f"    {lbl:<30} {d:.6f}")

    print(f"\n{'-'*92}")
    print("  Mann-Whitney U : baseline vs decodeur adouci")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups["baseline (decodeur original)"][metric], groups["decodeur adouci (test)"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print("  Mann-Whitney U : decodeur adouci vs NSGA-III")
    print(f"{'-'*92}")
    for metric in ("HV", "GD", "IGD", "Spacing"):
        a, b = groups["decodeur adouci (test)"][metric], groups["NSGA-III (reference)"][metric]
        if len(a) < 2 or len(b) < 2:
            print(f"  {metric:<8} : pas assez de runs valides")
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "significatif (p<0.05)" if p < 0.05 else "non significatif"
        print(f"  {metric:<8} U={u:.1f}  p={p:.6f}  -> {sig}")

    print(f"\n{'-'*92}")
    print("  Temps moyen par run")
    print(f"{'-'*92}")
    for lbl in ("baseline (decodeur original)", "decodeur adouci (test)"):
        s = _stats(groups[lbl]["elapsed_s"])
        print(f"    {lbl:<30} mean={s['mean']:.1f}s")

    print(f"\n{'='*92}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test de decodeur adouci (softmax) — QINSGA-III")
    parser.add_argument("--instance", default="100", choices=["3", "5", "15", "25", "30", "40", "100"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--gen",   type=int, default=300)
    parser.add_argument("--pop",   type=int, default=POP_SIZE)
    parser.add_argument("--temp",  type=float, default=DEFAULT_TEMP)
    args = parser.parse_args()
    run_comparison(args.instance, args.seeds, args.gen, args.pop, args.temp)
