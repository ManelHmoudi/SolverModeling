"""End-to-end tests for run_nsga3's repair_final_front parameter, on the
tiny 3-client instance with a minimal generation count -- fast enough to
run as a regular test, still exercising the real pymoo search loop."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Solvers.NSGA3 import main as nsga3_main

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TINY_INSTANCE = os.path.join(PROJECT_DIR, "data", "instance_3_clients.json")


def test_run_nsga3_repair_final_front_defaults_to_true(tmp_path, monkeypatch):
    monkeypatch.setattr(nsga3_main, "_CHROM_CACHE_PATH", str(tmp_path / "nsga3_chromosomes.json"))

    data = nsga3_main.run_nsga3(data_path=_TINY_INSTANCE, n_gen=2, n_runs=1)

    assert data["runs"][0]["meta"]["repair_final_front"] is True


def test_run_nsga3_repair_final_front_false_runs_and_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(nsga3_main, "_CHROM_CACHE_PATH", str(tmp_path / "nsga3_chromosomes.json"))

    data = nsga3_main.run_nsga3(
        data_path=_TINY_INSTANCE, n_gen=2, n_runs=1, repair_final_front=False,
    )

    assert data["runs"][0]["meta"]["repair_final_front"] is False


def test_run_nsga3_repair_final_front_true_runs_and_is_recorded(tmp_path, monkeypatch):
    cache_path = tmp_path / "nsga3_chromosomes.json"
    monkeypatch.setattr(nsga3_main, "_CHROM_CACHE_PATH", str(cache_path))

    data = nsga3_main.run_nsga3(
        data_path=_TINY_INSTANCE, n_gen=2, n_runs=1, repair_final_front=True,
    )

    assert data["runs"][0]["meta"]["repair_final_front"] is True
    with open(cache_path, encoding="utf-8") as f:
        cache = json.load(f)
    assert cache["runs"][0]["meta_base"]["repair_final_front"] is True


def test_render_from_instance_reapplies_stored_repair_flag(tmp_path, monkeypatch):
    cache_path = tmp_path / "nsga3_chromosomes.json"
    monkeypatch.setattr(nsga3_main, "_CHROM_CACHE_PATH", str(cache_path))

    nsga3_main.run_nsga3(
        data_path=_TINY_INSTANCE, n_gen=2, n_runs=1, repair_final_front=True,
    )
    refreshed = nsga3_main.render_from_instance(_TINY_INSTANCE)

    assert refreshed["runs"][0]["meta"]["repair_final_front"] is True
