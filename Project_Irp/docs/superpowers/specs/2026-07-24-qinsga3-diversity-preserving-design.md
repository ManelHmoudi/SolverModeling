# QINSGA3 per-niche diversity-preserving operator — design

## Context

The elitist per-niche guide (`docs/superpowers/specs/2026-07-24-qinsga3-elitist-guide-design.md`,
implemented and committed but **not adopted**) fixed DTLZ1's seed-to-seed
variance but regressed DTLZ3 by 3-5x (mean/worst/std) in a 5-run ablation —
see the "Piste explorée et écartée" note in
`validation/dtlz/results/qinsga3/DTLZ_results_summary_qinsga3.md`. Hypothesis:
on DTLZ3's strongly deceptive multimodal landscape, a niche's guide can lock
onto a locally-dominant-but-globally-suboptimal solution, and once the whole
niche's population converges toward it, there is nothing left in the
population diverse enough to discover a solution that would dominate it and
evict it from the archive — a textbook premature-convergence trap.

The user asked for a fix grounded in one specific, verifiable paper rather
than an invented combination. After confirming the primary literature
source was inaccessible (paywalled), the user supplied the full text of
**Tayarani-N & Akbarzadeh-T (2014), "Improvement of the performance of the
Quantum-inspired Evolutionary Algorithms: structures, population,
operators", Evolutionary Intelligence 7:219–239**, §5, "Diversity Preserving
operator" (DPQiEA) — which is exactly this failure mode's documented fix in
the QEA literature: individuals that have converged and clustered around the
same local optimum are detected, the fittest is kept, and the rest are
reinitialized to break them out of that basin of attraction.

## Goals

- Implement DPQiEA's mechanism (§5, eq. 11-14) for QINSGA3, adapted to its
  continuous θ-encoding and per-reference-direction niche structure, as a
  new operator applied after mutation in both generational loops
  (`QINSGA3/algorithm.py::run_qinsga3` and
  `validation/algorithms/qinsga3/core.py::run_qinsga3_generic`).
- Port the paper's formulas with the minimum adaptation the domain change
  requires (see "Formula mapping" below) — every deviation from the paper
  is justified explicitly, not invented.
- Validate with the same 5-run DTLZ1+DTLZ3 ablation methodology already
  established for every other QINSGA3 tuning decision in this project,
  before considering a full 30-run campaign.

## Non-goals (explicit user constraints)

- **No change to the benchmark protocol**: DTLZ1-7/MaF1-7 problems, the 30
  shared seeds, Das-Dennis reference directions, population size, and the
  Cui et al. (2025) generation-count budget stay exactly as they are.
- **No change to the "quantum spirit"**: the rotation gate (`rotate()`),
  the θ-encoding and measurement (`measure()`), and the existing
  crossover/mutation operators in `QINSGA3/chromosome.py` are untouched.
  This is a new, additive operator — it does not replace or alter any
  existing quantum mechanism.
- **No change to the parameters shared with NSGA-III for comparison**:
  `P_CROSS=1.0`, `ETA_CROSS=20`, `p_mut=1/D` (aligned to Cui et al. 2025)
  stay exactly as adopted. This operator's new parameters (`GAMMA`,
  `DELTA`, `T_STAGNATION`) are independent additions with no NSGA-III
  equivalent, same status as `MIGRATION_PERIOD`/`N_MIGRATE`/`NOISE_SCALE`
  already are.
- **No change to `_select_guides`, `_archive_update`, or the archive
  itself.** The paper's operator only touches the population's
  q-individuals, never the tracked "best observed solution" — ported
  verbatim: *"we only reinitialize the q-individuals, not the best observed
  solutions"* (§5). The archive keeps self-correcting via its own existing
  dominance check in `_archive_update` if a freshly-reset niche discovers
  something better — no new archive-side mechanism is added (this replaces
  and drops the earlier, non-literature-grounded "archive cooldown" idea
  from the previous design pass).

## Formula mapping (paper → QINSGA3)

| Paper (§5, binary QiEA) | QINSGA3 (continuous θ ∈ [0, π/2]) | Status |
|---|---|---|
| Convergence, eq. 11: `(1/m) Σ_k \|1 − 2\|α_ik\|²\| > γ` | `(1/n_genes) Σ_k \|cos(2θ_ik)\| > γ` | **Exact port.** QINSGA3 already uses `\|α_ik\|² = cos²(θ_ik)` (`chromosome.py::measure`), and `1 − 2cos²θ ≡ −cos(2θ)` algebraically — no interpretation involved. |
| Reinitialization value, eq. 14: `q_ik ← 1/√2` | `θ_ik ← π/4` | **Exact port.** QINSGA3 already calls `θ = π/4` "maximum superposition" at population init (`chromosome.py::__init__`) — the same state in its own convention, not an approximation. |
| Keep-best/reinit-rest structure | Same | **Exact port.** No change to the core rule. |
| "Best observed solution" `b_i`/`B_i` per neighbourhood, tracked separately from the q-individual (never reinitialized) | The niche's elitist guide θ (from the already-implemented, un-adopted Task 1-4 work) | **Direct mapping**, not an adaptation — the guide already *is* QINSGA3's "best solution known for this neighbourhood," playing exactly the role `b_i` plays in the paper. |
| Similarity, eq. 12: `(1/m) Σ_k \|x_ik − x_jk\| < δ` on **observed binary bits** | `(1/n_genes) Σ_k \|θ_ik − θ_jk\| / (π/2) < δ` on **θ, normalised to [0,1]** | **Necessary adaptation.** The paper's domain is binary (Hamming-style distance on 0/1 bits); QINSGA3's genes are continuous angles. Normalising by the domain width (π/2) keeps the same [0,1]-ish scale the paper's own bit-distance has, so the paper's empirical δ values remain a meaningful starting point rather than needing a re-derivation from scratch. |
| "Enough time" (§5 step 6, eq. 13): `b_i^{t−T} = b_i^t`, i.e. the neighbourhood's best hasn't changed in T iterations | The niche's guide θ (elitist, Task 1-4) hasn't changed in `T_STAGNATION` generations | **Direct mapping.** Same reasoning as the `b_i` row above — this is not a new invention, it is eq. 13 applied to QINSGA3's own already-existing per-niche "best," which happens to already be tracked as the elitist guide. |
| "Neighbourhood" (paper invents Ring/Cellular/Star/etc. topologies because combinatorial problems have no natural grouping) | The reference-direction **niche** (`assoc`), already computed every generation for guide selection | **Necessary adaptation, but not an invention** — QINSGA3 already has a real, problem-driven grouping the paper's own topologies were a proxy for. Using it directly is more faithful to the paper's *intent* (group individuals exploiting the same region) than reusing an arbitrary combinatorial-domain topology would be. |
| Best-of-cluster selection (paper: "keep the best q-individual", fitness in the single-objective sense) | Individual closest to the niche's reference ray (same criterion `_select_guides` already uses to pick a niche's own front representative) | **Necessary adaptation** — QINSGA3 is many-objective, so "best" is operationalised the same way the rest of the algorithm already operationalises "best in niche," not a new criterion. |

## Design

### New state (loop-owned, mirrors the existing archive-list pattern)

- `niche_guide_history: dict[int, np.ndarray]` — last generation's guide θ
  per niche id.
- `niche_stagnation: dict[int, int]` — consecutive-generations-unchanged
  counter per niche id.

Both initialised empty before the generation loop, updated once per
generation, in both `run_qinsga3` and `run_qinsga3_generic`.

### New pure helpers (`QINSGA3/algorithm.py`, same style as `_select_guides` etc.)

- `_update_niche_stagnation(assoc, guides_theta, history, counters) ->
  (new_history, new_counters, stagnant_niches)` — eq. 13: compares this
  generation's per-niche guide θ (already computed by `_select_guides`) to
  the stored value; increments or resets the counter; returns the set of
  niches whose counter has reached `T_STAGNATION`.
- `_diversity_preserve_mask(assoc, qpop_theta, F_norm, ref_dirs,
  stagnant_niches, gamma, delta) -> reset_mask` — eq. 11 (convergence) +
  eq. 12 (similarity clustering, adapted) restricted to `stagnant_niches`;
  within each converged-and-similar cluster, keeps the member closest to
  the reference ray, marks the rest in `reset_mask` (shape `(pop_size,)`).

### Loop wiring

In both `run_qinsga3` and `run_qinsga3_generic`, after `qpop.mutate(...)`
and before the migration step (mirrors the paper's flowchart: the operator
is the last per-generation touch on individuals before the next
measurement — migration is QINSGA3's own later addition and stays after,
consistent with "archive stays untouched by this operator"):

```
niche_guide_history, niche_stagnation, stagnant = _update_niche_stagnation(
    assoc, guides_theta, niche_guide_history, niche_stagnation)

if T_STAGNATION > 0 and stagnant:
    reset_mask = _diversity_preserve_mask(
        assoc, qpop.theta, F_norm, ref_dirs, stagnant, GAMMA, DELTA)
    qpop.theta[reset_mask] = np.pi / 4
```

(`T_STAGNATION > 0` as the enable/disable idiom, matching the codebase's
existing `if p_cross > 0.0` / `if migration_period > 0` convention — no
separate boolean flag.)

### New parameters

`GAMMA` (convergence threshold), `DELTA` (similarity threshold),
`T_STAGNATION` (generations of unchanged guide before triggering) — new,
additive, QINSGA3-only constants with no NSGA-III equivalent, added
alongside the existing `MIGRATION_PERIOD`/`N_MIGRATE`/`NOISE_SCALE` in
`validation/algorithms/qinsga3/runner.py` (benchmark) and
`QINSGA3/main.py` (production defaults). Starting values from the paper's
own empirically-found good ranges (Table 4): `GAMMA=0.99`,
`T_STAGNATION=5` (paper's best values cluster at 2 or 5),
`DELTA` starting at `0.1` (paper found this the most problem-sensitive
parameter — flagged as the primary ablation target, exactly like
`noise_scale` was for the earlier tuning pass).

### Testing

New unit tests in `QINSGA3/test_algorithm.py` for `_update_niche_stagnation`
and `_diversity_preserve_mask`, following the same small-synthetic-array
style as the Task 1-4 tests: a niche whose guide is unchanged for exactly
`T_STAGNATION` generations triggers; one generation short does not; a
"converged" individual (by eq. 11) that is NOT similar (eq. 12) to any
other converged individual in its niche is not reset; a converged+similar
cluster keeps the member closest to the reference ray and resets the rest
to exactly `π/4`.

### Empirical validation

Same protocol as the elitist-guide ablation: 5 runs on DTLZ1 + DTLZ3 (same
5 seeds), compared against the current adopted config (crossover aligned +
noise=0 + migration 5/20 — the elitist guide from Task 1-4 stays
un-adopted, so this operator is validated **on top of the currently
adopted config, not on top of the un-adopted elitist guide**, to isolate
its own effect). If DTLZ3 improves without regressing DTLZ1 (or others,
spot-checked): proceed to a full 30-run campaign; otherwise stop and
document the outcome the same way the elitist-guide ablation was
documented, per the project's established practice of recording
explored-and-rejected paths.

## Open risk / thing to watch

The paper's own experiments (Table 4) show `DELTA`'s best value varies
hugely by problem (0.01 to 1) — there is no reason to expect a single
`DELTA` to be simultaneously right for DTLZ1's shape and DTLZ3's, mirroring
the earlier `noise_scale` dilemma (DTLZ7 vs DTLZ3). If the 5-run ablation
shows a similar opposed-trends pattern, that is a documented limitation to
report, not a signal to keep tuning `DELTA` indefinitely.
