"""Unit tests for validate_irp parsers and distance function."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from validate_irp import (
    compute_distance,
    parse_dat_file,
    parse_reference,
    get_ref_path,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

ABS1N5_DAT = """\
6  3  289
   1     154.0     417.0         510         193       .03
   2     172.0     334.0  130  195    0   65       .02
   3     267.0      87.0   70  105    0   35       .03
   4     148.0     433.0   58  116    0   58       .03
   5     355.0     444.0   48   72    0   24       .02
   6      38.0     152.0   11   22    0   11       .02
"""

SOL_LOW = "TOTAL COSTS: 1235.92\ni[0][0]: 510\n"
SOL_HIGH = "TOTAL COSTS: 2108.34\ni[0][0]: 510\n"

# ── Distance ──────────────────────────────────────────────────────────────────

def test_compute_distance_depot_to_r3():
    # depot(154,417) → retailer3(148,433): sqrt(36+256)=17.088 → 17
    assert compute_distance(154.0, 417.0, 148.0, 433.0) == 17

def test_compute_distance_zero():
    assert compute_distance(100.0, 200.0, 100.0, 200.0) == 0

def test_compute_distance_rounding_up():
    # sqrt(0.5^2+0.5^2) = 0.7071 → NINT = 1
    assert compute_distance(0.0, 0.0, 0.5, 0.5) == 1

def test_compute_distance_rounding_down():
    # sqrt(0.3^2+0.3^2) = 0.4243 → NINT = 0
    assert compute_distance(0.0, 0.0, 0.3, 0.3) == 0

# ── parse_dat_file ─────────────────────────────────────────────────────────────

def test_parse_dat_header(tmp_path):
    f = tmp_path / "abs1n5.dat"
    f.write_text(ABS1N5_DAT)
    inst = parse_dat_file(str(f))
    assert inst.name == "abs1n5"
    assert inst.H == 3
    assert inst.C == 289.0
    assert len(inst.retailers) == 5

def test_parse_dat_depot(tmp_path):
    f = tmp_path / "abs1n5.dat"
    f.write_text(ABS1N5_DAT)
    depot = parse_dat_file(str(f)).depot
    assert depot.x  == 154.0
    assert depot.y  == 417.0
    assert depot.I0 == 510.0
    assert depot.r  == 193.0
    assert depot.h  == pytest.approx(0.03)

def test_parse_dat_retailer_0(tmp_path):
    f = tmp_path / "abs1n5.dat"
    f.write_text(ABS1N5_DAT)
    r = parse_dat_file(str(f)).retailers[0]   # file node 2 → internal index 1
    assert r.x  == 172.0
    assert r.y  == 334.0
    assert r.I0 == 130.0
    assert r.U  == 195.0
    assert r.L  == 0.0
    assert r.r  == 65.0
    assert r.h  == pytest.approx(0.02)

def test_parse_dat_retailer_last(tmp_path):
    f = tmp_path / "abs1n5.dat"
    f.write_text(ABS1N5_DAT)
    r = parse_dat_file(str(f)).retailers[4]   # file node 6 → internal index 5
    assert r.x  == 38.0
    assert r.y  == 152.0
    assert r.I0 == 11.0
    assert r.U  == 22.0
    assert r.L  == 0.0
    assert r.r  == 11.0
    assert r.h  == pytest.approx(0.02)

# ── parse_reference ────────────────────────────────────────────────────────────

def test_parse_reference_lowcost(tmp_path):
    f = tmp_path / "sol.dat"
    f.write_text(SOL_LOW)
    assert parse_reference(str(f)) == pytest.approx(1235.92)

def test_parse_reference_highcost(tmp_path):
    f = tmp_path / "sol.dat"
    f.write_text(SOL_HIGH)
    assert parse_reference(str(f)) == pytest.approx(2108.34)

def test_parse_reference_missing_raises(tmp_path):
    f = tmp_path / "empty.dat"
    f.write_text("no costs here\n")
    with pytest.raises(ValueError, match="TOTAL COSTS"):
        parse_reference(str(f))

# ── get_ref_path ───────────────────────────────────────────────────────────────

def test_get_ref_path_lowcost():
    path = get_ref_path("/ref", "lowcost", 1, 5)
    assert path == os.path.join(
        "/ref", "instances_low_cost_H3",
        "absH3low1n5K=1", "absH3low1n5K=1_solution.dat"
    )

def test_get_ref_path_highcost():
    path = get_ref_path("/ref", "highcost", 3, 10)
    assert path == os.path.join(
        "/ref", "instances_high_cost_H3",
        "absH3high3n10K=1", "absH3high3n10K=1_solution.dat"
    )

def test_get_ref_path_variant5_n5_low():
    path = get_ref_path("/ref", "lowcost", 5, 5)
    assert path == os.path.join(
        "/ref", "instances_low_cost_H3",
        "absH3low5n5K=1", "absH3low5n5K=1_solution.dat"
    )


# ── Integration tests (require CPLEX + external data files) ───────────────────

from validate_irp import validate_instance

_INST_LOW  = r"C:\Users\Mariem\OneDrive\Bureau\Master\MPIRP\IRP-Research\DataSets\Archetti2007\Instances_lowcost_H3"
_INST_HIGH = r"C:\Users\Mariem\OneDrive\Bureau\Master\MPIRP\IRP-Research\DataSets\Archetti2007\Instances_highcost_H3"
_REF_DIR   = (
    r"C:\Users\Mariem\OneDrive\Bureau\Master\MPIRP\IRP-Research"
    r"\Results_Reference\Results_Iterative_Matheuristic_Vadseth_2021\Run1"
)

_data_present = pytest.mark.skipif(
    not os.path.isdir(_INST_LOW),
    reason="Archetti instance files not accessible"
)


@_data_present
def test_solve_abs1n5_lowcost():
    """Full CPLEX solve: abs1n5 lowcost.

    The MIP finds the provably optimal solution; the Vadseth (2021) reference is
    a matheuristic and may be sub-optimal.  We therefore assert:
      - a feasible solution was found (f1_cplex != "")
      - f1_reference parses correctly
      - f1_cplex <= f1_reference * 1.001  (our MIP must not be worse than ref by >0.1%)
    Being strictly better than the heuristic reference is expected and correct.
    """
    inst_path = os.path.join(_INST_LOW, "abs1n5.dat")
    ref_path  = get_ref_path(_REF_DIR, "lowcost", 1, 5)

    row = validate_instance(inst_path, ref_path, timelimit=300, cost_type="lowcost")

    assert row["f1_reference"] == pytest.approx(1235.92, rel=1e-6)
    assert row["f1_cplex"]     != ""
    f1_cplex = float(row["f1_cplex"])
    f1_ref   = float(row["f1_reference"])
    assert f1_cplex <= f1_ref * 1.001, (
        f"MIP solution {f1_cplex} is more than 0.1% worse than heuristic ref {f1_ref} "
        f"— check objective formula"
    )


@_data_present
def test_solve_abs1n5_highcost():
    """Full CPLEX solve: abs1n5 highcost.

    Same rationale as lowcost: MIP is exact, reference is heuristic.
    Assert f1_cplex <= f1_reference * 1.001.
    """
    inst_path = os.path.join(_INST_HIGH, "abs1n5.dat")
    ref_path  = get_ref_path(_REF_DIR, "highcost", 1, 5)

    row = validate_instance(inst_path, ref_path, timelimit=300, cost_type="highcost")

    assert row["f1_reference"] == pytest.approx(2108.34, rel=1e-6)
    assert row["f1_cplex"]     != ""
    f1_cplex = float(row["f1_cplex"])
    f1_ref   = float(row["f1_reference"])
    assert f1_cplex <= f1_ref * 1.001, (
        f"MIP solution {f1_cplex} is more than 0.1% worse than heuristic ref {f1_ref} "
        f"— check t0_offset or distance formula"
    )
