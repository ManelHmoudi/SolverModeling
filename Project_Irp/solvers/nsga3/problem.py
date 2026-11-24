"""pymoo ElementwiseProblem for the many-objective IRP.

Chromosome: gene[l_idx * n_periods + t_idx] = quantity delivered to client l in period t.

Objectives (all minimised): f1 logistics cost, f2 CO2, f3 travel time, f4 working capital.
Hard constraints via out["G"] (g <= 0 = feasible): C13 tau_return window,
C14 delivery deadlines, C6 depot stock ceiling (safety net — the decoder
already forces enough shipment to respect it structurally in the common case).
"""

from pymoo.core.problem import ElementwiseProblem

from .decoder   import decode_chromosome, build_routes
from .evaluator import compute_f1, compute_f2, compute_f3, compute_f4


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

        n_prio = len(clients)
        xl += [0.0] * n_prio
        xu += [1.0] * n_prio

        super().__init__(
            n_var        = len(clients) * len(T) + n_prio,
            n_obj        = 4,
            n_ieq_constr = 2 * len(T) + len(clients) * len(T) + 2 * len(T),
            xl           = xl,
            xu           = xu,
        )
        self.sets_   = sets_
        self.params_ = params_

    def _evaluate(self, x, out, *args, **kwargs):
        quantities, priorities = decode_chromosome(x, self.sets_)
        route_result = build_routes(quantities, self.sets_, self.params_, priorities)

        out["F"] = [
            compute_f1(route_result, self.sets_, self.params_),
            compute_f2(route_result, self.sets_, self.params_),
            compute_f3(route_result, self.sets_, self.params_),
            compute_f4(route_result, self.sets_, self.params_),
        ]

        clients = self.sets_["clients"]
        T       = self.sets_["T"]
        q_lt    = self.params_["q_lt"]
        tau_min  = self.params_["tau_min"]
        tau_max  = self.params_["tau_max"]
        I_max_f  = self.params_["I_O_max_frigo"]
        I_max_nf = self.params_["I_O_max_nonfrigo"]
        actual      = route_result["actual_qty"]
        depot_stock = route_result["depot_stock"]

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

        # C6 upper bound safety net — the decoder already forces enough
        # shipment to respect I_O_max structurally; this catches the residual
        # case where remaining demand/truck capacity can't absorb the surplus.
        for t in T:
            G.append(depot_stock[t]["frigo"]    - I_max_f)
            G.append(depot_stock[t]["nonfrigo"] - I_max_nf)

        out["G"] = G
