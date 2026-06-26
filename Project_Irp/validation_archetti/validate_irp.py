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

from docplex.mp.model import Model

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
    n_total = int(toks[0])   # depot + retailers
    H       = int(toks[1])
    C       = float(toks[2])

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
