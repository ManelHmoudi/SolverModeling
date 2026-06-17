"""
pymoo ElementwiseProblem for the many-objective IRP.

Chromosome: gene[l_idx * n_periods + t_idx] = quantity delivered to client l in period t.

Objectives (all minimised by pymoo):
  F[0] = f1   logistics cost
  F[1] = f2   CO2 emissions
  F[2] = f3   total travel time
  F[3] = f4   working capital (minimised — lower BFR = less capital tied up)

Hard constraints via out["G"] (g <= 0 = feasible):
  C13 — tau_return[t] within [tau_min, tau_max] for every period t
  C14 — cumulative delivered[l, t] >= cumulative demand[l, t] for all l, t
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

        # Priority genes — one per client, ∈ [0, 1]
        # Higher value → client visited earlier by the NN decoder
        n_prio = len(clients)
        xl += [0.0] * n_prio
        xu += [1.0] * n_prio

        n_C13 = 2 * len(T)                    # lower + upper bound per period
        n_C14 = len(clients) * len(T)          # cumulative deadline per (l, t)

        super().__init__(
            n_var        = len(clients) * len(T) + n_prio,
            n_obj        = 4,
            n_ieq_constr = n_C13 + n_C14,
            xl           = xl,
            xu           = xu,
        )
        self.sets_   = sets_
        self.params_ = params_

    def _evaluate(self, x, out, *args, **kwargs):
        quantities, priorities = decode_chromosome(x, self.sets_)
        route_result = build_routes(quantities, self.sets_, self.params_, priorities)

        f1 = compute_f1(route_result, self.sets_, self.params_)
        f2 = compute_f2(route_result, self.sets_, self.params_)
        f3 = compute_f3(route_result, self.sets_, self.params_)
        f4 = compute_f4(route_result, self.sets_, self.params_)

        out["F"] = [f1, f2, f3, f4]

        clients  = self.sets_["clients"]
        T        = self.sets_["T"]
        q_lt     = self.params_["q_lt"]
        tau_min  = self.params_["tau_min"]
        tau_max  = self.params_["tau_max"]
        actual   = route_result["actual_qty"]

        G = []

        # C13: global tour return window — hard constraint
        for t in T:
            ret = route_result["tau_return"].get(t, 0.0)
            G.append(ret - tau_max)   # tau_return[t] <= tau_max
            G.append(tau_min - ret)   # tau_return[t] >= tau_min

        # C14: cumulative delivery deadline — hard constraint
        # For each client l and period t: sum_{td<=t} delivered[l,td] >= sum_{td<=t} demand[l,td]
        for l in clients:
            cum_del = cum_dem = 0
            for t in T:
                cum_del += actual.get((l, t), 0)
                cum_dem += q_lt[l, t]
                G.append(cum_dem - cum_del)   # g <= 0 iff cum_del >= cum_dem

        out["G"] = G