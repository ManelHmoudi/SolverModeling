"""
pymoo ElementwiseProblem for the many-objective IRP.

Chromosome: gene[l_idx * n_periods + t_idx] = quantity delivered to client l in period t.

Objectives (all minimised by pymoo):
  F[0] = f1   logistics cost
  F[1] = f2   CO2 emissions
  F[2] = f3   total travel time
  F[3] = -f4  working capital (negated so pymoo minimises)

Hard constraints via out["G"] (g <= 0 = feasible):
  C13 — tau_return[t] within [tau_min, tau_max] for every period t
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
                # lower: enforce period-1 demand on first period, 0 otherwise
                xl.append(float(q_lt[l, T_first]) if t == T_first else 0.0)
                # upper: can anticipate full horizon demand in one period
                xu.append(total_l)

        super().__init__(
            n_var        = len(clients) * len(T),
            n_obj        = 4,
            n_ieq_constr = 2 * len(T),   # C13: lower + upper bound per period
            xl           = xl,
            xu           = xu,
        )
        self.sets_   = sets_
        self.params_ = params_

    def _evaluate(self, x, out, *args, **kwargs):
        quantities   = decode_chromosome(x, self.sets_)
        route_result = build_routes(quantities, self.sets_, self.params_)

        f1 = compute_f1(route_result, self.sets_, self.params_)
        f2 = compute_f2(route_result, self.sets_, self.params_)
        f3 = compute_f3(route_result, self.sets_, self.params_)
        f4 = compute_f4(route_result, self.sets_, self.params_)

        out["F"] = [f1, f2, f3, -f4]

        # C13: global tour return window — hard constraint
        tau_min = self.params_["tau_min"]
        tau_max = self.params_["tau_max"]
        G = []
        for t in self.sets_["T"]:
            ret = route_result["tau_return"].get(t, 0.0)
            G.append(ret - tau_max)   # tau_return[t] <= tau_max
            G.append(tau_min - ret)   # tau_return[t] >= tau_min
        out["G"] = G