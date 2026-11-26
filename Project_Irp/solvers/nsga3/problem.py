"""pymoo ElementwiseProblem for the many-objective IRP (model v2).

Chromosome: see decoder.py (quantity genes, priority genes, order genes).

Objectives (all minimised): f1 logistics cost, f2 CO2, f3 longest routes,
f4 peak net working capital -- docs/model_formulation.md §6.

Hard constraints via out["G"] (g <= 0 = feasible), built by
build_g_constraints(), the single implementation shared with QI-NSGA-III.
"""

from pymoo.core.problem import ElementwiseProblem

from .decoder   import decode_chromosome, build_routes, n_genes, PRODUCT_CLASSES
from .evaluator import compute_f1, compute_f2, compute_f3, compute_f4, route_duration


def n_constraints(sets_):
    nC, nT = len(sets_["clients"]), len(sets_["T"])
    return nT + 2 * nC * nT + 4 * nT + len(PRODUCT_CLASSES)


def build_g_constraints(route_result, sets_, params_):
    """Constraint vector (g <= 0 feasible), fixed length n_constraints(sets_):
      - shift: latest return <= tau_max                                     (C15)
      - demand never late, per client, period AND class                     (C9, C10)
      - depot stock between I_min and I_max at the end of each period       (C5)
      - cyclic plan: end-of-horizon stock == initial stock per class        (C7)
    """
    clients, T = sets_["clients"], sets_["T"]
    q, cold = params_["q_lt"], params_["requires_cold"]
    stock, orders = route_result["depot_stock"], route_result["orders"]

    G = []
    for t in T:
        # recomputed from the paths, so it stays exact after any repair move
        rets = [route_duration(info["path"], k, params_)
                for k, info in route_result["routes_data"].get(t, {}).items() if len(info["path"]) > 2]
        G.append((max(rets) if rets else 0.0) - params_["tau_max"])

    served = {}
    for t, routes in route_result["routes_data"].items():
        for k, info in routes.items():
            p = "frigo" if k in params_["frigo_trucks"] else "nonfrigo"
            for l_str, qty in info["qty"].items():
                served[int(l_str), t, p] = served.get((int(l_str), t, p), 0) + qty
    for l in clients:
        for p in PRODUCT_CLASSES:
            cum_del = cum_dem = 0
            for t in T:
                cum_del += served.get((l, t, p), 0)
                cum_dem += q[l, t] if cold[l, t] == (p == "frigo") else 0
                G.append(cum_dem - cum_del)

    for p in PRODUCT_CLASSES:
        for t in T:
            G.append(stock[t][p] - params_[f"I_O_max_{p}"])
            G.append(params_[f"I_O_min_{p}"] - stock[t][p])
    for p in PRODUCT_CLASSES:
        G.append(abs(stock[T[-1]][p] - params_[f"I_O_init_{p}"]) - 1e-6)
    return G


def evaluate_route_result(route_result, sets_, params_):
    """(F, G) for an already-decoded (and possibly repaired) solution."""
    F = [
        compute_f1(route_result, sets_, params_),
        compute_f2(route_result, sets_, params_),
        compute_f3(route_result, sets_, params_),
        compute_f4(route_result, sets_, params_),
    ]
    return F, build_g_constraints(route_result, sets_, params_)


def decode_full(x, sets_, params_):
    """Chromosome → route_result (decoder only, no repair)."""
    quantities, priorities, order_genes = decode_chromosome(x, sets_)
    return build_routes(quantities, sets_, params_, priorities, order_genes)


class IRPProblem(ElementwiseProblem):

    def __init__(self, sets_, params_):
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

        xl += [0.0] * len(clients)                       # priority genes
        xu += [1.0] * len(clients)
        xl += [0.0] * (len(PRODUCT_CLASSES) * len(T))    # order genes
        xu += [1.0] * (len(PRODUCT_CLASSES) * len(T))
        assert len(xl) == n_genes(sets_)

        super().__init__(
            n_var        = n_genes(sets_),
            n_obj        = 4,
            n_ieq_constr = n_constraints(sets_),
            xl           = xl,
            xu           = xu,
        )
        self.sets_   = sets_
        self.params_ = params_

    def _evaluate(self, x, out, *args, **kwargs):
        route_result = decode_full(x, self.sets_, self.params_)
        out["F"], out["G"] = evaluate_route_result(route_result, self.sets_, self.params_)
