"""End-to-end tests for run_qinsga3_solver's repair_final_front parameter --
covers the bug where the solver computed a correctly-repaired pareto_F
internally but then discarded it, re-decoding the report/cache from raw
chromosomes without ever applying the 2-opt repair. Uses the tiny 3-client
instance with a minimal generation count."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Solvers.QINSGA3 import main as qinsga3_main

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TINY_INSTANCE = os.path.join(PROJECT_DIR, "data", "instance_3_clients.json")


def test_run_qinsga3_solver_repair_final_front_defaults_to_true(tmp_path, monkeypatch):
    monkeypatch.setattr(qinsga3_main, "_CHROM_CACHE_PATH", str(tmp_path / "qinsga3_chromosomes.json"))

    data = qinsga3_main.run_qinsga3_solver(data_path=_TINY_INSTANCE, n_gen=2, n_runs=1)

    assert data["runs"][0]["meta"]["repair_final_front"] is True


def test_run_qinsga3_solver_repair_final_front_false_is_recorded(tmp_path, monkeypatch):
    cache_path = tmp_path / "qinsga3_chromosomes.json"
    monkeypatch.setattr(qinsga3_main, "_CHROM_CACHE_PATH", str(cache_path))

    data = qinsga3_main.run_qinsga3_solver(
        data_path=_TINY_INSTANCE, n_gen=2, n_runs=1, repair_final_front=False,
    )

    assert data["runs"][0]["meta"]["repair_final_front"] is False
    with open(cache_path, encoding="utf-8") as f:
        cache = json.load(f)
    assert cache["runs"][0]["meta_base"]["repair_final_front"] is False


def test_render_from_instance_reapplies_stored_repair_flag(tmp_path, monkeypatch):
    cache_path = tmp_path / "qinsga3_chromosomes.json"
    monkeypatch.setattr(qinsga3_main, "_CHROM_CACHE_PATH", str(cache_path))

    qinsga3_main.run_qinsga3_solver(
        data_path=_TINY_INSTANCE, n_gen=2, n_runs=1, repair_final_front=False,
    )
    refreshed = qinsga3_main.render_from_instance(_TINY_INSTANCE)

    assert refreshed["runs"][0]["meta"]["repair_final_front"] is False
