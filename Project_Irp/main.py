"""
Many-Objective Inventory Routing Problem (IRP)
===============================================
Objectives:
    f1 - Logistics cost        (transport + storage + time-window penalties)
    f2 - CO2 emissions         (CMEM model, Bektas & Laporte 2011)
    f3 - Total travel time     (sum of arc travel times)
    f4 - Working capital (BFR) (stock value + receivables - payables)

Network:
    G = (N, A)
    Depot       = node 0
    Clients     = {1, 2, 3}
    Destination = node 4
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from docplex.mp.model import Model
from models.parametres  import sets_, params_, N, A, T, M, clients, stock_nodes
from models.variables   import build_variables
from models.objectives  import build_all_objectives
from models.constraints import add_all_constraints


# =============================================================================
# MODEL AND DECISION VARIABLES
# =============================================================================

mdl   = Model(name="IRP_ManyObjective")
vars_ = build_variables(mdl, N, A, T, M, clients, stock_nodes)


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

objectives = build_all_objectives(mdl, vars_, sets_, params_)

# Inject objective expressions into params_ so budget constraints can reference them
params_["objectives"] = objectives


# =============================================================================
# CONSTRAINTS
# =============================================================================

add_all_constraints(mdl, vars_, sets_, params_)


# =============================================================================
# MODEL SUMMARY
# =============================================================================

print(f"Model        : {mdl.name}")
print(f"Nodes        : {len(N)}  (depot + {len(clients)} customers + destination)")
print(f"Stock nodes  : {stock_nodes}")
print(f"Arcs         : {len(A)}")
print(f"Periods      : {len(T)}")
print(f"Vehicles     : {len(M)}")
print(f"Variables    : {mdl.number_of_variables}")
print(f"Constraints  : {mdl.number_of_constraints}")


# =============================================================================
# CALIBRATION — solve each objective independently to suggest bound values
# =============================================================================

def test_objective(name, expr):
    """Minimise a single objective and report the optimal value with routing details."""
    mdl.minimize(expr)
    sol = mdl.solve(log_output=False)
    if sol:
        print(f"  {name} = {expr.solution_value:.4f}")
        used = [
            (i, j, t, k)
            for (i, j, t, k) in vars_["x"]
            if vars_["x"][i, j, t, k].solution_value > 0.5
        ]
        print(f"    Arcs used  : {used}")
        deliveries = {
            (l, t): round(vars_["q_prime"][l, t].solution_value, 6)
            for (l, t) in vars_["q_prime"]
        }
        print(f"    Deliveries : {deliveries}")
        return expr.solution_value
    else:
        print(f"  {name} : infeasible -- {mdl.solve_details}")
        return None


print("\n── Calibration f1 (logistics cost) ──")
f1_val = test_objective("f1", objectives["f1"])
if f1_val:
    print(f"    → Suggested C_max : {f1_val * 1.2:.4f}")

print("\n── Calibration f2 (CO2 emissions) ──")
f2_val = test_objective("f2", objectives["f2"])
if f2_val:
    print(f"    → Suggested E_max : {f2_val * 1.2:.4f}")

print("\n── Calibration f3 (travel time) ──")
f3_val = test_objective("f3", objectives["f3"])
if f3_val:
    print(f"    → Suggested T_max : {f3_val * 1.2:.4f}")

print("\n── Calibration f4 (BFR) ──")
f4_val = test_objective("f4", objectives["f4"])
if f4_val:
    sub = objectives["f4_sub"]
    print(f"    → Suggested B     : {f4_val * 1.2:.4f}")
    print(f"    Breakdown :")
    print(f"      Stock value   = {sub['stock_value'].solution_value:.4f}")
    print(f"      Receivables   = {sub['receivables'].solution_value:.4f}")
    print(f"      Payables (-)  = {sub['payables'].solution_value:.4f}")