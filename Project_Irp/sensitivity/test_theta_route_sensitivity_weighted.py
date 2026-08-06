"""H2 properly weighted across the WHOLE run, not just 4 spot alpha values.

Context: test_theta_route_sensitivity.py already tested H2 (professor's
hypothesis: "a small theta variation on the IRP might not produce any real
route change") at 4 fixed alpha values spanning the production range
(alpha_min=0.001*pi .. alpha_max=0.10*pi) and found it holds only in a narrow
window (alpha near alpha_min, i.e. the very end of a run) -- 19.2% no-change
at alpha=0.00314, dropping to ~0% by alpha=0.03142. That is a qualitative
"narrow window" conclusion from 4 isolated points; this script makes it
quantitative by sampling alpha at every point along the ACTUAL linear decay
schedule production uses (alpha(g) = alpha_min + (alpha_max-alpha_min)*(1-g/max_gen),
Solvers/QINSGA3/algorithm.py::run_qinsga3) and computing the single number
that answers "H2, integrated over the whole run": what fraction of all
rotation events across a full 300-generation run leave the route unchanged?

Usage: python -m sensitivity.test_theta_route_sensitivity_weighted
"""
import hashlib
import json
import os
import sys

PROJECT_DIR = r"c:\Users\Mariem\OneDrive\Bureau\SolverModeling\Project_Irp"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np

from models.parametres        import load_instance
from Solvers.NSGA3.decoder     import decode_chromosome, build_routes
from Solvers.NSGA3.evaluator   import compute_f1, compute_f2, compute_f3, compute_f4
from Solvers.NSGA3.problem     import IRPProblem
from Solvers.QINSGA3.algorithm import _encode_theta

INSTANCE   = "100"
SEEDS      = [42, 137, 271, 491, 613]
MAX_GEN    = 300
N_SAMPLES  = 31          # generations sampled: 0, 10, 20, ..., 300
ALPHA_MAX  = 0.10 * np.pi
ALPHA_MIN  = 0.001 * np.pi
RNG_SEED   = 12345

_NSGA3_CACHE = os.path.join(PROJECT_DIR, "Solvers", "NSGA3", "nsga3_chromosomes.json")


def _route_signature(route_result) -> str:
    parts = []
    for t in sorted(route_result["routes_data"].keys(), key=str):
        for k in sorted(route_result["routes_data"][t].keys(), key=str):
            info = route_result["routes_data"][t][k]
            path_str = "-".join(str(n) for n in info["path"])
            qty_str  = ",".join(f"{l}:{q}" for l, q in sorted(info["qty"].items()))
            parts.append(f"{t}|{k}|{path_str}|{qty_str}")
    return hashlib.sha1("||".join(parts).encode()).hexdigest()


def _decode_one(x, sets_, params_):
    quantities, priorities = decode_chromosome(x, sets_)
    rr = build_routes(quantities, sets_, params_, priorities)
    F = np.array([
        compute_f1(rr, sets_, params_), compute_f2(rr, sets_, params_),
        compute_f3(rr, sets_, params_), compute_f4(rr, sets_, params_),
    ])
    return _route_signature(rr), F


def main():
    data_path = os.path.join(PROJECT_DIR, "data", f"instance_{INSTANCE}_clients.json")
    sets_, params_ = load_instance(data_path)
    problem = IRPProblem(sets_, params_)
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)

    with open(_NSGA3_CACHE, encoding="utf-8") as f:
        cache = json.load(f)
    by_seed = {r["seed"]: r for r in cache["runs"]}
    baseline_X = []
    for s in SEEDS:
        baseline_X.extend(by_seed[s]["chromosomes"])
    baseline_X = np.array(baseline_X, dtype=float)
    N = len(baseline_X)
    print(f"Baseline: {N} solutions reelles (NSGA-III cache, {len(SEEDS)} seeds)")

    theta = _encode_theta(baseline_X, xl, xu)
    rng = np.random.default_rng(RNG_SEED)

    print("\nDecodage des baselines (routes de reference)...")
    base_sig = []
    for i in range(N):
        sig, _ = _decode_one(baseline_X[i], sets_, params_)
        base_sig.append(sig)

    gens = np.linspace(0, MAX_GEN, N_SAMPLES)
    alphas = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * (1.0 - gens / MAX_GEN)

    print("\n" + "=" * 96)
    print("  H2 PONDERE SUR LE CALENDRIER ALPHA REEL (schedule lineaire, 0..300 generations)")
    print("=" * 96)
    print(f"  {N_SAMPLES} generations echantillonnees (tous les {MAX_GEN // (N_SAMPLES - 1)} gens), "
          f"N baselines={N}")
    print("=" * 96)

    pct_identical_per_gen = []
    for gen, alpha in zip(gens, alphas):
        guide_idx = rng.integers(0, N, size=N)
        same = guide_idx == np.arange(N)
        guide_idx[same] = (guide_idx[same] + 1) % N
        theta_guide = theta[guide_idx]

        diff = theta_guide - theta
        theta_pert = theta + alpha * np.tanh(diff / (np.pi / 8.0))
        theta_pert = np.clip(theta_pert, 0.0, np.pi / 2.0)

        p = np.cos(theta_pert) ** 2
        X_pert = np.clip(xl + p * (xu - xl), xl, xu)

        n_identical = 0
        for i in range(N):
            sig, _ = _decode_one(X_pert[i], sets_, params_)
            if sig == base_sig[i]:
                n_identical += 1
        pct = 100.0 * n_identical / N
        pct_identical_per_gen.append(pct)
        print(f"  gen={gen:6.1f}  alpha={alpha:.5f}  routes identiques={pct:5.1f}%")

    pct_identical_per_gen = np.array(pct_identical_per_gen)
    # generations are uniformly spaced -> simple mean = generation-weighted average
    weighted_avg = pct_identical_per_gen.mean()

    print(f"\n{'='*96}")
    print("  RESULTAT PONDERE SUR TOUT LE RUN")
    print(f"{'='*96}")
    print(f"  Moyenne (ponderee par generation) du taux de routes inchangees : {weighted_avg:.2f}%")
    print(f"  Max observe (fin de run, alpha proche de alpha_min)            : {pct_identical_per_gen.max():.1f}%")
    print(f"  Fraction des {N_SAMPLES} points d'echantillonnage avec >5% inchange : "
          f"{100.0*float((pct_identical_per_gen > 5.0).mean()):.1f}%")
    print(f"{'='*96}")
    print("\nINTERPRETATION")
    print("  Si la moyenne ponderee est faible (proche de 0%), H2 n'explique pas une part")
    print("  significative du budget de rotation gaspille sur l'ensemble du run -- la fenetre")
    print("  etroite en fin de run (alpha faible) ne pese pas assez dans la moyenne generationnelle")
    print("  pour etre le facteur dominant.")
    print(f"{'='*96}\n")


if __name__ == "__main__":
    main()
