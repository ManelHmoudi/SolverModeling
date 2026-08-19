"""Shared report-building helpers used by both NSGA3 and QINSGA3.

Extracted here so that QINSGA3 does not need to import private symbols
from NSGA3.main, keeping the two solver modules fully decoupled.
"""

import numpy as np

from .decoder   import decode_chromosome, build_routes
from .evaluator import compute_f1, compute_f2, compute_f3, compute_f4_detail
from .metrics   import compute_pareto_metrics, build_empirical_reference_front
from .report    import compute_node_positions


def _build_delivery_rows(route_result, sets_, params_):
    """Per-(client, period) delivery rows with cumulative coverage for the report."""
    deliveries = []
    for l in sets_["clients"]:
        cum_recu = cum_dem = 0
        last_k   = -1
        for t in sets_["T"]:
            recu = int(route_result["actual_qty"].get((l, t), 0))
            dem  = int(params_["q_lt"].get((l, t), 0))
            if recu > 0:
                last_k = route_result["truck_assign"].get((l, t), -1)
            cum_recu += recu
            cum_dem  += dem
            deliveries.append({
                "l":        l,
                "t":        t,
                "k":        route_result["truck_assign"].get((l, t), last_k),
                "recu":     recu,
                "dem":      dem,
                "cum_recu": cum_recu,
                "cum_dem":  cum_dem,
                "balance":  cum_recu - cum_dem,
            })
    return deliveries


def _evaluate_pareto(pareto_X, sets_, params_, meta_base, repair: bool = False,
                      use_or_opt: bool = False, use_single_relocation: bool = False,
                      use_two_opt: bool = False, use_inter_route_relocate: bool = False,
                      use_route_swap: bool = False, use_delivery_shift: bool = True):
    """Evaluate Pareto chromosomes and return the run data dict.

    Called after a fresh solver run and on every report refresh
    (with potentially updated instance data). When repair=True, each
    decoded route is passed through QINSGA3's post-decode 2-opt local
    search (Solvers/QINSGA3/repair.py, Baldwinian: the chromosome itself
    is unaffected, only the fitness/routes reported) before objectives
    are computed -- shared by both NSGA3 and QINSGA3's report pipelines so
    a repair_final_front flag means the same thing for either algorithm.

    use_or_opt (default False, only meaningful when repair=True): forwarded
    to _repair_route_result -- see its own use_or_opt docstring. Shared by
    both algorithms' report pipelines for the same reason repair itself is,
    so an Or-opt fairness comparison between NSGA-III and QI-NSGA-III (see
    sensitivity/compare_2opt_fairness.py's own --use-or-opt flag) applies
    identically to both sides.

    use_single_relocation (default False, only meaningful when repair=True):
    forwarded to _repair_route_result the same way -- see its own
    use_single_relocation docstring.

    use_two_opt (default True, only meaningful when repair=True): forwarded
    to _repair_route_result the same way -- see its own use_two_opt
    docstring.

    use_inter_route_relocate (default False, only meaningful when
    repair=True): forwarded to _repair_route_result the same way -- see
    its own use_inter_route_relocate docstring.

    use_route_swap (default False, only meaningful when repair=True):
    forwarded to _repair_route_result the same way -- see its own
    use_route_swap docstring.

    use_delivery_shift (default False, only meaningful when repair=True):
    forwarded to _repair_route_result the same way -- see its own
    use_delivery_shift docstring.
    """
    solutions = []
    for i, chromosome in enumerate(pareto_X):
        quantities, priorities = decode_chromosome(chromosome, sets_)
        route_result = build_routes(quantities, sets_, params_, priorities)
        if repair:
            from Solvers.QINSGA3.repair import _repair_route_result
            route_result = _repair_route_result(route_result, sets_, params_, use_or_opt=use_or_opt,
                                                 use_single_relocation=use_single_relocation,
                                                 use_two_opt=use_two_opt,
                                                 use_inter_route_relocate=use_inter_route_relocate,
                                                 use_route_swap=use_route_swap,
                                                 use_delivery_shift=use_delivery_shift)

        f1         = compute_f1(route_result, sets_, params_)
        f2         = compute_f2(route_result, sets_, params_)
        f3         = compute_f3(route_result, sets_, params_)
        f4, f4_sub = compute_f4_detail(route_result, sets_, params_)

        routes_report = {}
        for t in sets_["T"]:
            trucks_t = [
                {"k": k, "path": info["path"], "qty": info["qty"]}
                for k, info in route_result["routes_data"].get(t, {}).items()
            ]
            tau_ret = route_result["tau_return"].get(t, 0.0)
            routes_report[str(t)] = {
                "trucks":     trucks_t,
                "R_frigo":    params_["R_frigo"].get(t, 0.0),
                "R_nonfrigo": params_["R_nonfrigo"].get(t, 0.0),
                "tau_return": round(tau_ret, 4),
                "shipped":    sum(route_result["actual_qty"].get((l, t), 0)
                                  for l in sets_["clients"]),
            }

        solutions.append({
            "id":         i,
            "objectives": {
                "f1": round(f1, 4),
                "f2": round(f2, 4),
                "f3": round(f3, 4),
                "f4": round(f4, 4),
            },
            "routes":      routes_report,
            "depot_stock": {str(t): route_result["depot_stock"][t] for t in sets_["T"]},
            "deliveries":  _build_delivery_rows(route_result, sets_, params_),
            "bfr_sub":     f4_sub,
        })

    F = np.array([[s["objectives"]["f1"], s["objectives"]["f2"],
                   s["objectives"]["f3"], s["objectives"]["f4"]]
                  for s in solutions])

    if repair and len(solutions) > 0:
        # The 2-opt repair above optimises f1 only, per individual -- it can
        # improve one solution's f1 while worsening its f2/f3, which can
        # make it newly dominated by another solution in the same front
        # (confirmed on a real cached front: 40/40 non-dominated before
        # repair, only 26/40 after). Re-filter to non-dominated before
        # reporting/scoring, or HV/GD/IGD/Spacing get computed over a set
        # that isn't actually a Pareto front any more.
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        nd_idx = NonDominatedSorting().do(F)[0]
        if len(nd_idx) < len(solutions):
            solutions = [solutions[i] for i in nd_idx]
            for new_id, s in enumerate(solutions):
                s["id"] = new_id
            F = F[nd_idx]

    quality = compute_pareto_metrics(F)

    return {
        "meta": {
            **meta_base,
            "n_nodes":           len(sets_["N"]),
            "n_clients":         len(sets_["clients"]),
            "n_periods":         len(sets_["T"]),
            "n_vehicles":        len(sets_["M"]),
            "n_pareto":          len(solutions),
            "node_positions":    compute_node_positions(sets_["N"]),
            "I_O_init_frigo":    params_["I_O_init_frigo"],
            "I_O_init_nonfrigo": params_["I_O_init_nonfrigo"],
            "quality":           quality,
        },
        "solutions": solutions,
    }


def _build_report_data(runs_data):
    """Aggregate per-run data into the final report dict.

    Metrics (HV, GD, IGD, Spacing) are recomputed using a GLOBAL ideal/nadir
    shared across all runs so per-run values are directly comparable.
    """
    all_F = np.vstack([
        np.array([[s["objectives"]["f1"], s["objectives"]["f2"],
                   s["objectives"]["f3"], s["objectives"]["f4"]]
                  for s in r["solutions"]])
        for r in runs_data
    ])
    g_ideal = all_F.min(axis=0)
    g_nadir = all_F.max(axis=0)
    # PF_ref = ND(union of every solution across every run) -- empirical
    # reference front for GD/IGD instead of a Das-Dennis reference-direction
    # grid. See Solvers/NSGA3/metrics.py's build_empirical_reference_front.
    # Each run is scored against a LEAVE-ONE-RUN-OUT variant (built from
    # every OTHER run only) below, never this exact shared front, so a run's
    # own solutions can never contribute to the reference it is measured
    # against (which would let it trivially score a zero self-distance).
    per_run_F = [
        np.array([[s["objectives"]["f1"], s["objectives"]["f2"],
                   s["objectives"]["f3"], s["objectives"]["f4"]]
                  for s in r["solutions"]])
        for r in runs_data
    ]

    for idx, r in enumerate(runs_data):
        F_run = per_run_F[idx]
        other_F = [F2 for j, F2 in enumerate(per_run_F) if j != idx]
        pf_ref_minus_r = build_empirical_reference_front(np.vstack(other_F)) if other_F else None
        old_q = r["meta"]["quality"]
        new_q = compute_pareto_metrics(F_run, g_ideal, g_nadir, reference_front=pf_ref_minus_r)
        new_q["ideal"] = old_q.get("ideal")
        new_q["nadir"] = old_q.get("nadir")
        r["meta"]["quality"] = new_q

    def _stats(vals):
        arr = [v for v in vals if v is not None]
        if not arr:
            return {"mean": None, "std": None, "min": None, "max": None}
        a = np.array(arr, dtype=float)
        return {
            "mean": round(float(a.mean()), 6),
            "std":  round(float(a.std()),  6),
            "min":  round(float(a.min()),  6),
            "max":  round(float(a.max()),  6),
        }

    hvs      = [r["meta"]["quality"].get("HV")      for r in runs_data]
    gds      = [r["meta"]["quality"].get("GD")      for r in runs_data]
    igds     = [r["meta"]["quality"].get("IGD")     for r in runs_data]
    spacings = [r["meta"]["quality"].get("Spacing") for r in runs_data]
    nps      = [float(r["meta"]["n_pareto"])         for r in runs_data]

    stability = {
        "HV":       _stats(hvs),
        "GD":       _stats(gds),
        "IGD":      _stats(igds),
        "Spacing":  _stats(spacings),
        "n_pareto": _stats(nps),
    }

    first = runs_data[0]["meta"]
    outer_meta = {
        "instance":          first["instance"],
        "n_nodes":           first["n_nodes"],
        "n_clients":         first["n_clients"],
        "n_periods":         first["n_periods"],
        "n_vehicles":        first["n_vehicles"],
        "pop_size":          first["pop_size"],
        "n_gen":             first["n_gen"],
        "crossover_prob":    first["crossover_prob"],
        "mutation_prob":     first["mutation_prob"],
        "n_runs":            first.get("n_runs", len(runs_data)),
        "n_completed":       first.get("n_completed", len(runs_data)),
        "node_positions":    first["node_positions"],
        "I_O_init_frigo":    first["I_O_init_frigo"],
        "I_O_init_nonfrigo": first["I_O_init_nonfrigo"],
    }
    for _extra in ("algorithm", "alpha_max", "alpha_min", "eta_cross", "repair_final_front"):
        if _extra in first:
            outer_meta[_extra] = first[_extra]

    return {
        "meta":      outer_meta,
        "runs":      runs_data,
        "stability": stability,
    }
