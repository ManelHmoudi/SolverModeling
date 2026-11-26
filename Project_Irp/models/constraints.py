"""
Many-Objective Inventory Routing Problem (IRP)
===============================================
Constraints C1-C17 -- docs/model_formulation.md §7.
(C18 and C19 are built with f4 and f3 in objectives.py.)
(C18, the peak working-capital epigraph, is built with f4 in objectives.py.)
"""

from .objectives import delivered
from .variables import PRODUCT_CLASSES


def _net_delivery(mdl, f, N, l, t, k):
    return (mdl.sum(f[i, l, t, k] for i in N if i != l)
            - mdl.sum(f[l, j, t, k] for j in N if j != l))


def _visit(mdl, x, N, l, t, k):
    return mdl.sum(x[i, l, t, k] for i in N if i != l)


# ── Routing and capacity (C1-C4) ──────────────────────────────────────────────

def add_routing_constraints(mdl, vars_, sets_, params_):
    x, f = vars_["x"], vars_["f"]
    N, A, T, M, O = sets_["N"], sets_["A"], sets_["T"], sets_["M"], sets_["O"]
    for t in T:
        for k in M:
            # C1: a vehicle leaves the depot at most once per period
            mdl.add_constraint(mdl.sum(x[O, j, t, k] for j in N if j != O) <= 1, ctname=f"C1_t{t}_k{k}")
            # C2: flow conservation at every client (implies the return to the depot),
            # and at most one visit per client, vehicle and period. The visit limit is
            # already implied by the exact arrival times (a revisit would need a time
            # cycle); it is stated explicitly for clarity and a tighter relaxation.
            for j in N:
                if j != O:
                    inflow = mdl.sum(x[i, j, t, k] for i in N if i != j)
                    mdl.add_constraint(inflow == mdl.sum(x[j, i, t, k] for i in N if i != j),
                                       ctname=f"C2_j{j}_t{t}_k{k}")
                    mdl.add_constraint(inflow <= 1, ctname=f"C2_visit_j{j}_t{t}_k{k}")
            for (i, j) in A:
                if j == O:
                    # C4: vehicles return empty
                    mdl.add_constraint(f[i, j, t, k] == 0, ctname=f"C4_{i}_t{t}_k{k}")
                else:
                    # C3: load only on travelled arcs, within capacity
                    mdl.add_constraint(f[i, j, t, k] <= params_["Q"][k] * x[i, j, t, k],
                                       ctname=f"C3_{i}_{j}_t{t}_k{k}")


# ── Depot inventory and supplier orders (C5-C8) ───────────────────────────────

def add_depot_constraints(mdl, vars_, sets_, params_):
    q_prime, r, z = vars_["q_prime"], vars_["r"], vars_["z"]
    T, clients = sets_["T"], sets_["clients"]
    cold = params_["requires_cold"]
    stock = {"frigo": vars_["I_O_frigo"], "nonfrigo": vars_["I_O_nonfrigo"]}

    for p in PRODUCT_CLASSES:
        I = stock[p]
        is_cold = p == "frigo"
        for t in T:
            prev = I[t - 1] if t != T[0] else params_[f"I_O_init_{p}"]
            # C5: safety stock and depot capacity
            mdl.add_constraint(I[t] >= params_[f"I_O_min_{p}"], ctname=f"C5_min_{p}_t{t}")
            mdl.add_constraint(I[t] <= params_[f"I_O_max_{p}"], ctname=f"C5_max_{p}_t{t}")
            # C6: balance = previous stock + supplier order - shipments of class p
            shipped = mdl.sum(q_prime[l, t, td] for l in clients for td in T
                              if td >= t and cold[l, td] == is_cold)
            mdl.add_constraint(I[t] == prev + r[p, t] - shipped, ctname=f"C6_{p}_t{t}")
            # C8: an order is placed (fixed cost) whenever something is ordered
            mdl.add_constraint(r[p, t] <= params_[f"R_bigM_{p}"] * z[p, t], ctname=f"C8_link_{p}_t{t}")
        # C7: cyclic plan -- the horizon ends with the stock it started with, so
        # the plan repeats and the cyclic financial accounting is consistent
        mdl.add_constraint(I[T[-1]] == params_[f"I_O_init_{p}"], ctname=f"C7_cycle_{p}")


# ── Demand (C9) ───────────────────────────────────────────────────────────────

def add_demand_constraints(mdl, vars_, sets_, params_):
    T, clients = sets_["T"], sets_["clients"]
    for l in clients:
        for t in T:
            # C9: every demand is met on time or early, never late
            mdl.add_constraint(mdl.sum(vars_["q_prime"][l, tp, t] for tp in T if tp <= t)
                               == params_["q_lt"][l, t], ctname=f"C9_l{l}_t{t}")


# ── Deliveries and vehicle compatibility (C10-C11) ────────────────────────────

def add_delivery_constraints(mdl, vars_, sets_, params_):
    x, f, q_prime = vars_["x"], vars_["f"], vars_["q_prime"]
    N, T, M, clients = sets_["N"], sets_["T"], sets_["M"], sets_["clients"]
    cold, frigo = params_["requires_cold"], params_["frigo_trucks"]
    groups = (([k for k in M if k in frigo], True, "cold"), ([k for k in M if k not in frigo], False, "amb"))
    for l in clients:
        for t in T:
            # C10: cold products only on refrigerated trucks, ambient products only on standard trucks
            for trucks, is_cold, tag in groups:
                mdl.add_constraint(
                    mdl.sum(_net_delivery(mdl, f, N, l, t, k) for k in trucks)
                    == mdl.sum(q_prime[l, t, td] for td in T if td >= t and cold[l, td] == is_cold),
                    ctname=f"C10_{tag}_l{l}_t{t}")
            # C11 (second part): at most one truck per product class serves l in t (no split
            # delivery within a class; a client with cold and ambient demand gets one truck of each)
            for trucks, _, tag in groups:
                if trucks:
                    mdl.add_constraint(mdl.sum(_visit(mdl, x, N, l, t, k) for k in trucks) <= 1,
                                       ctname=f"C11_nosplit_{tag}_l{l}_t{t}")
            # C11: a visit delivers at least one unit (no empty visits; also rules out subtours)
            for k in M:
                mdl.add_constraint(_net_delivery(mdl, f, N, l, t, k) >= _visit(mdl, x, N, l, t, k),
                                   ctname=f"C11_l{l}_t{t}_k{k}")


# ── Time: per vehicle, no idle waiting (C12-C16) ──────────────────────────────

def add_time_constraints(mdl, vars_, sets_, params_):
    x, a, e, w1, w2 = (vars_[k] for k in ("x", "a", "e", "w1", "w2"))
    N, A, T, M, O, clients = (sets_[k] for k in ("N", "A", "T", "M", "O", "clients"))
    s, d, v = params_["s"], params_["d"], params_["v"]
    tau_max = params_["tau_max"]

    for t in T:
        for k in M:
            # C12: departure from the depot at time 0
            mdl.add_constraint(a[O, t, k] == 0, ctname=f"C12_t{t}_k{k}")
            for (i, j) in A:
                travel = s[i] + d[i, j] / v[k]  # for i = O, s[O] is the loading time at the depot
                target = e[t, k] if j == O else a[j, t, k]
                tag = "C14" if j == O else "C13"
                # C13 / C14: arrival (resp. return) = departure + service + travel on used arcs.
                # Tightest valid big-M with every time in [0, tau_max]:
                #   ">=" side: tau_max + travel  (target >= a_i - tau_max always holds)
                #   "<=" side: tau_max           (target - a_i - travel <= tau_max always holds)
                mdl.add_constraint(target >= a[i, t, k] + travel - (tau_max + travel) * (1 - x[i, j, t, k]),
                                   ctname=f"{tag}_lb_{i}_{j}_t{t}_k{k}")
                mdl.add_constraint(target <= a[i, t, k] + travel + tau_max * (1 - x[i, j, t, k]),
                                   ctname=f"{tag}_ub_{i}_{j}_t{t}_k{k}")
            # C15: shift length -- e_t,k <= tau_max for a used vehicle, e_t,k = 0 for an unused one
            mdl.add_constraint(e[t, k] <= tau_max * mdl.sum(x[O, j, t, k] for j in N if j != O),
                               ctname=f"C15_t{t}_k{k}")
            # C16: soft time windows, per vehicle visit
            # Tightest valid big-M: ET for earliness (a >= 0), max(0, tau_max - LT) for lateness (a <= tau_max)
            for l in clients:
                visit = _visit(mdl, x, N, l, t, k)
                ET, LT = params_["ET"][l, t], params_["LT"][l, t]
                mdl.add_constraint(w1[l, t, k] >= ET - a[l, t, k] - ET * (1 - visit),
                                   ctname=f"C16_early_l{l}_t{t}_k{k}")
                mdl.add_constraint(w2[l, t, k] >= a[l, t, k] - LT - max(0.0, tau_max - LT) * (1 - visit),
                                   ctname=f"C16_late_l{l}_t{t}_k{k}")


# ── Symmetry breaking (C17) ───────────────────────────────────────────────────

def identical_vehicle_groups(sets_, params_):
    """Vehicles that are interchangeable: same class, capacity and speed."""
    groups = {}
    for k in sorted(sets_["M"]):
        key = (k in params_["frigo_trucks"], params_["Q"][k], params_["v"][k])
        groups.setdefault(key, []).append(k)
    return [g for g in groups.values() if len(g) > 1]


def add_symmetry_breaking(mdl, vars_, sets_, params_):
    x, N, T, O = vars_["x"], sets_["N"], sets_["T"], sets_["O"]
    for group in identical_vehicle_groups(sets_, params_):
        for k1, k2 in zip(group, group[1:]):
            for t in T:
                # C17: among identical vehicles, a higher index is used only if the lower one is
                mdl.add_constraint(mdl.sum(x[O, j, t, k1] for j in N if j != O)
                                   >= mdl.sum(x[O, j, t, k2] for j in N if j != O),
                                   ctname=f"C17_k{k1}_k{k2}_t{t}")


def add_all_constraints(mdl, vars_, sets_, params_):
    add_routing_constraints(mdl, vars_, sets_, params_)      # C1-C4
    add_depot_constraints(mdl, vars_, sets_, params_)        # C5-C8
    add_demand_constraints(mdl, vars_, sets_, params_)       # C9
    add_delivery_constraints(mdl, vars_, sets_, params_)     # C10-C11
    add_time_constraints(mdl, vars_, sets_, params_)         # C12-C16
    add_symmetry_breaking(mdl, vars_, sets_, params_)        # C17
