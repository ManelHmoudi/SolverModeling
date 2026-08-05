"""Local sensitivity test: does a small theta perturbation (the kind the
rotation gate actually applies) ever change the decoded route at all?
(Professor's hypothesis #2: QI-NSGA-III's rotation gate is suited to
continuous problems where x = f(theta) feeds objectives directly; on the
IRP, theta -> x -> decoder -> route means a small theta nudge might not
produce any real change in the route once the greedy nearest-neighbour
decoder and integer rounding are applied.)

Unlike the other diagnostics this session (which run the full 300-generation
algorithm), this test is cheap: it takes real, feasible baseline solutions
(NSGA-III's cached chromosomes, decoded via the SAME decoder QI-NSGA-III
uses -- Solvers/NSGA3/decoder.py), applies ONE realistic rotation step per
solution at several alpha magnitudes actually used in production
(alpha_min=0.001*pi up to alpha_max=0.10*pi), using another random baseline
as the "guide" (exactly QuantumPopulation.rotate()'s tanh formula), and
checks whether the decoded route changed at all.

Usage: python -m sensitivity.test_theta_route_sensitivity
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

INSTANCE  = "100"
SEEDS     = [42, 137, 271, 491, 613]
ALPHAS    = [0.001 * np.pi, 0.01 * np.pi, 0.05 * np.pi, 0.10 * np.pi]  # production alpha_min..alpha_max range
RNG_SEED  = 12345

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
    print(f"Baseline: {len(baseline_X)} solutions reelles (NSGA-III cache, {len(SEEDS)} seeds)")

    theta = _encode_theta(baseline_X, xl, xu)
    N = len(theta)

    rng = np.random.default_rng(RNG_SEED)

    print("\nDecodage des baselines (routes de reference)...")
    base_sig = []
    base_F   = []
    for i in range(N):
        sig, F = _decode_one(baseline_X[i], sets_, params_)
        base_sig.append(sig)
        base_F.append(F)
    base_F = np.array(base_F)

    print("\n" + "=" * 92)
    print("  SENSIBILITE LOCALE theta -> route (hypothese 2)")
    print("=" * 92)
    print(f"  Instance={INSTANCE} clients | N baselines={N} | alphas testes (rad) = {[round(a,5) for a in ALPHAS]}")
    print("  (alpha_min=0.001*pi=0.00314 ... alpha_max=0.10*pi=0.31416, plage de production)")
    print("=" * 92)

    for alpha in ALPHAS:
        guide_idx = rng.integers(0, N, size=N)
        # avoid self-guide
        same = guide_idx == np.arange(N)
        guide_idx[same] = (guide_idx[same] + 1) % N
        theta_guide = theta[guide_idx]

        diff = theta_guide - theta
        theta_pert = theta + alpha * np.tanh(diff / (np.pi / 8.0))
        theta_pert = np.clip(theta_pert, 0.0, np.pi / 2.0)

        p = np.cos(theta_pert) ** 2
        X_pert = np.clip(xl + p * (xu - xl), xl, xu)

        n_identical = 0
        dF_changed  = []
        dX_mean = float(np.abs(X_pert - baseline_X).mean())

        for i in range(N):
            sig, F = _decode_one(X_pert[i], sets_, params_)
            if sig == base_sig[i]:
                n_identical += 1
            else:
                dF_changed.append(np.linalg.norm(F - base_F[i]))

        pct_identical = 100.0 * n_identical / N
        print(f"\n  alpha={alpha:.5f} rad ({alpha/np.pi:.4f}*pi)")
        print(f"    |Delta X| moyen (espace reel)        : {dX_mean:.4f}")
        print(f"    Routes IDENTIQUES apres perturbation : {n_identical}/{N} ({pct_identical:.1f}%)")
        if dF_changed:
            dF_arr = np.array(dF_changed)
            print(f"    Quand la route change, |Delta F|     : mean={dF_arr.mean():.2f}  median={np.median(dF_arr):.2f}  min={dF_arr.min():.2f}")

    print(f"\n{'='*92}")
    print("  INTERPRETATION")
    print(f"{'='*92}")
    print("  Si le % de routes identiques reste eleve meme pour alpha_max (debut de run),")
    print("  l'hypothese 2 est CONFIRMEE : une part significative des pas de rotation ne")
    print("  produit aucun changement reel de tournee -- l'effort de pilotage en theta-space")
    print("  est en partie gaspille par le decodeur. Si le % identique est bas (~0%) meme a")
    print("  alpha_min (fin de run), l'hypothese 2 est REFUTEE : le decodeur est sensible a")
    print("  peu pres a tout pas de rotation, meme le plus petit.")
    print(f"{'='*92}\n")


if __name__ == "__main__":
    main()
