"""
Archetti 2007 IRP benchmark validation.

Solves f1 (transport + holding cost) mono-objective with DOcplex CPLEX.
Compares against Vadseth (2021) single-vehicle (K=1) reference solutions.

Usage:
    python validate_irp.py [--inst_lowcost DIR] [--inst_highcost DIR]
                           [--ref_dir DIR] [--n_sizes 5 10] [--variants 1 2 3 4 5]
                           [--timelimit 300] [--output validation_report.csv]
"""

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── Default paths ──────────────────────────────────────────────────────────────
INST_LOWCOST_DEFAULT = (
    r"C:\Users\Mariem\OneDrive\Bureau\Master\MPIRP\IRP-Research"
    r"\DataSets\Archetti2007\Instances_lowcost_H3"
)
INST_HIGHCOST_DEFAULT = (
    r"C:\Users\Mariem\OneDrive\Bureau\Master\MPIRP\IRP-Research"
    r"\DataSets\Archetti2007\Instances_highcost_H3"
)
REF_DIR_DEFAULT = (
    r"C:\Users\Mariem\OneDrive\Bureau\Master\MPIRP\IRP-Research"
    r"\Results_Reference\Results_Iterative_Matheuristic_Vadseth_2021\Run1"
)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Depot:
    x:  float
    y:  float
    I0: float   # initial inventory
    r:  float   # fixed production units received per period
    h:  float   # holding cost per unit per period


@dataclass
class Retailer:
    x:  float
    y:  float
    I0: float   # initial inventory
    U:  float   # maximum inventory (ML policy upper bound)
    L:  float   # minimum inventory (0 in all Archetti H3 instances)
    r:  float   # constant demand per period
    h:  float   # holding cost per unit per period


@dataclass
class Instance:
    name:      str
    H:         int           # planning horizon (number of periods)
    C:         float         # vehicle capacity
    depot:     Depot
    retailers: List[Retailer] = field(default_factory=list)
    # retailers[0] is internal node 1, retailers[i] is internal node i+1


# ── Pure functions ─────────────────────────────────────────────────────────────

def compute_distance(xi: float, yi: float, xj: float, yj: float) -> int:
    """NINT Euclidean distance used in Archetti (2007) instances."""
    return int(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2) + 0.5)


def parse_dat_file(path: str) -> Instance:
    """
    Parse an Archetti .dat instance file.

    Line 1:  n H C
    Line 2:  1  x  y  I0  r  h          (depot)
    Lines 3+: i  x  y  I0  U  L  r  h   (retailers)
    """
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    toks = lines[0].split()
    H = int(toks[1])
    C = float(toks[2])

    d       = lines[1].split()
    depot   = Depot(
        x=float(d[1]), y=float(d[2]),
        I0=float(d[3]), r=float(d[4]), h=float(d[5]),
    )

    retailers = []
    for ln in lines[2:]:
        t = ln.split()
        retailers.append(Retailer(
            x=float(t[1]), y=float(t[2]),
            I0=float(t[3]), U=float(t[4]),
            L=float(t[5]),  r=float(t[6]),
            h=float(t[7]),
        ))

    return Instance(name=name, H=H, C=C, depot=depot, retailers=retailers)


def parse_reference(path: str) -> float:
    """Read 'TOTAL COSTS: <value>' from a Vadseth *_solution.dat file."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("TOTAL COSTS:"):
                return float(line.split(":", 1)[1].strip())
    raise ValueError(f"'TOTAL COSTS' not found in {path}")


def get_ref_path(ref_dir: str, cost_type: str, variant: int, n: int) -> str:
    """
    Return the Vadseth K=1 solution file path for a given instance.

    cost_type: 'lowcost' or 'highcost'
    variant:   1–5
    n:         number of retailers (5 or 10)
    """
    tag    = "low"  if cost_type == "lowcost" else "high"
    subdir = "instances_low_cost_H3" if cost_type == "lowcost" else "instances_high_cost_H3"
    stem   = f"absH3{tag}{variant}n{n}K=1"
    return os.path.join(ref_dir, subdir, stem, f"{stem}_solution.dat")


# ── MIP model and orchestration (added in Task 2) ──────────────────────────────

def build_and_solve(
    inst: Instance,
    timelimit: int = 300,
) -> Tuple[Optional[float], float, str]:
    """
    Build and solve the Archetti IRP f1 MIP with DOcplex.

    Objective (MIP): transport_cost + Σ_{t=1..H} h_i·I[i,t]
    Post-solve:      f1_cplex = objective_value + t0_offset
    where            t0_offset = Σ_i h_i·I0_i  (holding cost at t=0, constant)

    Returns
    -------
    (f1_cplex, solve_time_s, status)
    f1_cplex is None if no feasible solution found.
    """
    from docplex.mp.model import Model

    n_ret    = len(inst.retailers)
    n        = n_ret + 1                          # total nodes; 0=depot, 1..n_ret=retailers
    nodes    = list(range(n))
    retailers = list(range(1, n))
    periods  = list(range(1, inst.H + 1))
    arcs     = [(i, j) for i in nodes for j in nodes if i != j]

    # Node coordinates: index 0 = depot, index i = retailers[i-1]
    coords = [(inst.depot.x, inst.depot.y)] + [
        (inst.retailers[i].x, inst.retailers[i].y) for i in range(n_ret)
    ]

    # Arc distances (integer, NINT)
    dist = {
        (i, j): compute_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
        for (i, j) in arcs
    }

    # Holding costs and initial inventories, indexed by internal node
    h_cost = {0: inst.depot.h}
    h_cost.update({i + 1: inst.retailers[i].h for i in range(n_ret)})

    I0 = {0: inst.depot.I0}
    I0.update({i + 1: inst.retailers[i].I0 for i in range(n_ret)})

    U      = {i + 1: inst.retailers[i].U for i in range(n_ret)}
    demand = {i + 1: inst.retailers[i].r for i in range(n_ret)}

    # t=0 holding cost — constant term, added to objective value after solving
    t0_offset = sum(h_cost[i] * I0[i] for i in nodes)

    # ── DOcplex model ──────────────────────────────────────────────────────────
    mdl = Model("Archetti_IRP_f1")
    mdl.parameters.timelimit                  = timelimit
    mdl.parameters.mip.tolerances.mipgap     = 1e-6
    mdl.parameters.mip.tolerances.absmipgap  = 1e-6
    mdl.set_log_output(False)

    # Decision variables
    x     = {(i, j, t): mdl.binary_var(name=f"x_{i}_{j}_{t}")
              for (i, j) in arcs for t in periods}
    q     = {(i, t): mdl.continuous_var(lb=0, name=f"q_{i}_{t}")
              for i in retailers for t in periods}
    I_var = {(i, t): mdl.continuous_var(lb=0, name=f"I_{i}_{t}")
              for i in nodes for t in periods}
    u_mtz = {(i, t): mdl.integer_var(lb=0, ub=n - 1, name=f"u_{i}_{t}")
              for i in retailers for t in periods}

    # Objective: transport + holding t=1..H  (t=0 added post-solve as constant)
    transport    = mdl.sum(dist[i, j] * x[i, j, t]
                           for (i, j) in arcs for t in periods)
    holding_t1_H = mdl.sum(h_cost[i] * I_var[i, t]
                            for i in nodes for t in periods)
    mdl.minimize(transport + holding_t1_H)

    # Constraint 1 — Depot inventory balance
    for t in periods:
        I_prev = I0[0] if t == 1 else I_var[0, t - 1]
        mdl.add_constraint(
            I_var[0, t] == I_prev + inst.depot.r - mdl.sum(q[i, t] for i in retailers),
            ctname=f"depot_balance_t{t}",
        )

    # Constraint 2 — Retailer inventory balance
    for i in retailers:
        for t in periods:
            I_prev = I0[i] if t == 1 else I_var[i, t - 1]
            mdl.add_constraint(
                I_var[i, t] == I_prev + q[i, t] - demand[i],
                ctname=f"retailer_balance_{i}_t{t}",
            )

    # Constraint 3 — ML (Maximum Level) policy
    for i in retailers:
        for t in periods:
            mdl.add_constraint(I_var[i, t] <= U[i], ctname=f"ml_{i}_t{t}")

    # Constraint 4 — Single vehicle: depot out/in-degree ≤ 1
    for t in periods:
        mdl.add_constraint(
            mdl.sum(x[0, j, t] for j in retailers) <= 1,
            ctname=f"depot_out_t{t}",
        )
        mdl.add_constraint(
            mdl.sum(x[j, 0, t] for j in retailers) <= 1,
            ctname=f"depot_in_t{t}",
        )

    # Constraint 5 — Flow conservation at every node
    for i in nodes:
        for t in periods:
            mdl.add_constraint(
                mdl.sum(x[i, j, t] for j in nodes if j != i)
                == mdl.sum(x[j, i, t] for j in nodes if j != i),
                ctname=f"flow_cons_{i}_t{t}",
            )

    # Constraint 6 — Each retailer visited at most once per period
    for i in retailers:
        for t in periods:
            mdl.add_constraint(
                mdl.sum(x[i, j, t] for j in nodes if j != i) <= 1,
                ctname=f"at_most_once_{i}_t{t}",
            )

    # Constraint 7 — MTZ subtour elimination (retailers only; depot is node 0)
    for i in retailers:
        for j in retailers:
            if i != j:
                for t in periods:
                    mdl.add_constraint(
                        u_mtz[i, t] - u_mtz[j, t] + n * x[i, j, t] <= n - 1,
                        ctname=f"mtz_{i}_{j}_t{t}",
                    )

    # Constraint 8 — Delivery only if retailer is visited
    for i in retailers:
        for t in periods:
            mdl.add_constraint(
                q[i, t] <= U[i] * mdl.sum(x[j, i, t] for j in nodes if j != i),
                ctname=f"link_delivery_{i}_t{t}",
            )

    # Constraint 9 — Vehicle capacity per period
    for t in periods:
        mdl.add_constraint(
            mdl.sum(q[i, t] for i in retailers) <= inst.C,
            ctname=f"capacity_t{t}",
        )

    # ── Solve ──────────────────────────────────────────────────────────────────
    t_start    = time.time()
    sol        = mdl.solve()
    solve_time = time.time() - t_start
    status     = mdl.solve_details.status

    if sol is None:
        return None, solve_time, status

    f1_cplex = mdl.objective_value + t0_offset
    return f1_cplex, solve_time, status


def validate_instance(
    inst_path: str,
    ref_path:  str,
    timelimit: int,
    cost_type: str,
) -> dict:
    """
    Parse, solve, and compare one instance.
    Returns a dict that maps directly to one CSV row.
    """
    inst   = parse_dat_file(inst_path)
    f1_ref = parse_reference(ref_path)

    f1_cplex, solve_time, status = build_and_solve(inst, timelimit)

    if f1_cplex is None:
        gap = ""
        f1_str = ""
    else:
        gap    = round(abs(f1_cplex - f1_ref) / f1_ref * 100.0, 4)
        f1_str = round(f1_cplex, 4)

    return {
        "instance":     inst.name,
        "n":            len(inst.retailers),
        "H":            inst.H,
        "cost_type":    cost_type,
        "f1_cplex":     f1_str,
        "f1_reference": round(f1_ref, 4),
        "gap_percent":  gap,
        "solve_time_s": round(solve_time, 2),
        "status":       status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archetti 2007 IRP validation — mono-objective f1 with DOcplex"
    )
    parser.add_argument("--inst_lowcost",  default=INST_LOWCOST_DEFAULT,
                        help="Path to Instances_lowcost_H3/")
    parser.add_argument("--inst_highcost", default=INST_HIGHCOST_DEFAULT,
                        help="Path to Instances_highcost_H3/")
    parser.add_argument("--ref_dir",       default=REF_DIR_DEFAULT,
                        help="Path to Vadseth2021/Run1/")
    parser.add_argument("--n_sizes",  nargs="+", type=int, default=[5, 10],
                        help="Retailer counts to solve (default: 5 10)")
    parser.add_argument("--variants", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        help="Instance variant indices (default: 1 2 3 4 5)")
    parser.add_argument("--timelimit", type=int, default=300,
                        help="CPLEX time limit per instance in seconds (default: 300)")
    parser.add_argument("--output",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "validation_report.csv"),
                        help="Output CSV path")
    args = parser.parse_args()

    fieldnames = [
        "instance", "n", "H", "cost_type",
        "f1_cplex", "f1_reference", "gap_percent",
        "solve_time_s", "status",
    ]
    rows = []

    for cost_type, inst_dir in [
        ("lowcost",  args.inst_lowcost),
        ("highcost", args.inst_highcost),
    ]:
        for n in args.n_sizes:
            for k in args.variants:
                inst_file = os.path.join(inst_dir, f"abs{k}n{n}.dat")
                ref_file  = get_ref_path(args.ref_dir, cost_type, k, n)

                if not os.path.exists(inst_file):
                    print(f"[SKIP] Instance not found: {inst_file}")
                    continue
                if not os.path.exists(ref_file):
                    print(f"[SKIP] Reference not found: {ref_file}")
                    continue

                label = f"{cost_type} abs{k}n{n}"
                print(f"[RUN ] {label} ...", end=" ", flush=True)
                try:
                    row = validate_instance(inst_file, ref_file, args.timelimit, cost_type)
                    rows.append(row)
                    if row["gap_percent"] != "":
                        gap_str = f"{row['gap_percent']:.4f}%"
                    else:
                        gap_str = "INFEASIBLE"
                    print(
                        f"f1={row['f1_cplex']}, ref={row['f1_reference']}, "
                        f"gap={gap_str}, t={row['solve_time_s']}s"
                    )
                except Exception as exc:
                    print(f"ERROR: {exc}")

    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    solved = [r for r in rows if r["gap_percent"] != ""]
    mean_gap = (
        sum(float(r["gap_percent"]) for r in solved) / len(solved)
        if solved else float("nan")
    )
    print(f"\n{'='*60}")
    print(f"Report saved : {args.output}")
    print(f"Instances    : {len(rows)} attempted, {len(solved)} solved")
    print(f"Mean gap     : {mean_gap:.4f}%")


if __name__ == "__main__":
    main()
