# QINSGA3 partial rotation by gene stability — screening design

## Context

`Solvers/IRP_results_summary.md`'s diagnostic chapter found QI-NSGA-III's
chromosome diversity on the real IRP is ~13x lower than NSGA-III's, driven by
the rotation gate pulling **every gene of every individual** toward its
niche's guide, every generation. Five "tuning" remedies (frequency, target,
magnitude of the same gate) were tried and rejected — none moved the
diversity number. One of them, `rotation_prob` (`sensitivity/
compare_rotation_prob.py`), masks rotation **per individual**, at random:
non-significant, diversity essentially unchanged (0.003984 → 0.003727). A
later remedy (G, post-decode 2-opt repair) found the actual gap driver is
decoder fragility, not diversity per se — the only remedy so far with a
significant HV/IGD gain.

This design tests a variant `rotation_prob` never covered: masking **per
gene, per individual**, selected by a stability criterion (already-close-to-
elites), rather than uniformly at random per individual. Different axis
(gene, not individual) and different selection rule (targeted, not random) —
worth a cheap check even though it belongs to the same "reduce what rotation
touches" family that has so far shown no significant effect on this
instance. Scope is deliberately minimal: a screening script, not a
production change — same protocol the project already uses to fast-reject
ideas before investing further (3 seeds, scale to 20 only on positive
signal).

## Goal

Check whether protecting already-divergent genes (the ones far from their
niche's elites) from the rotation pull — while still rotating genes that are
already converged/aligned — preserves more chromosome diversity and/or
improves HV/GD/IGD/Spacing vs. the current full-rotation baseline, without
degrading runtime.

## Design

### Selection criterion

Reuses `_elite_rms_distance` (`Solvers/QINSGA3/algorithm.py:875-908`)
unmodified: per individual, per gene, RMS distance from the individual's
current θ to its niche's archive elites (falls back to a single guide-point
distance, K=1, when the niche has no archive elites yet — same fallback
already used by `_chaotic_rotate`). Low value = gene already aligned with
elites ("stable"); high value = gene still far from consensus.

### Mask

Per individual, per generation: select the `frac` (default 0.15,
parametrised) fraction of genes with the **lowest** `elite_rms` — the most
stable ones. Only those genes receive this generation's rotation update; the
rest keep their pre-rotation θ value (shaped only by SBX/PM afterward, same
as `rotation_prob`'s un-rotated individuals).

```python
def _partial_rotation_mask(elite_rms: np.ndarray, frac: float) -> np.ndarray:
    pop_size, n_genes = elite_rms.shape
    k = max(1, int(round(frac * n_genes)))
    idx = np.argpartition(elite_rms, k - 1, axis=1)[:, :k]
    mask = np.zeros_like(elite_rms, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=1)
    return mask
```

Applied as a layer after the normal full rotation update, restoring
unselected genes to their pre-rotation value:

```python
theta_before = qpop.theta.copy()          # == theta_parent, pre-rotation
qpop.rotate(guides_theta, alpha)           # existing tanh gate, unchanged
mask = _partial_rotation_mask(elite_rms, frac)
qpop.theta = np.where(mask, qpop.theta, theta_before)
```

`elite_rms` itself is computed once per generation, before rotation, the
same way the existing chaotic-rotation branch already does (`arch_theta_arr`/
`arch_assoc_arr` from the archive, already resolved earlier in the loop).

### Scope

Standalone `sensitivity/compare_partial_rotation.py`, mirroring
`compare_rotation_prob.py`'s structure exactly: a private copy of the
generation loop (imports the same shared helpers from `Solvers/QINSGA3/
algorithm.py`), same comparison harness (shared ideal/nadir with the
`Solvers/NSGA3` cache, HV/GD/IGD/Spacing, chromosome diversity `Var(X_norm)`,
Mann-Whitney U vs. baseline and vs. NSGA-III). **No change to
`Solvers/QINSGA3/algorithm.py` or `chromosome.py`** — this stays ablation-only
until/unless a positive signal justifies promoting it to a production
parameter, same convention as every other remedy in this project.

Only the default `tanh` rotation gate is tested (matches `rotation_prob`'s
own baseline scope) — no interaction with `use_rqpso_rotation`/
`use_pso_rotation`/`use_chaotic_rotation` for this screening pass.

### CLI

`--instance` (default `100`), `--seeds` (default `[42, 137, 271]`), `--gen`
(default `300`), `--pop` (default `200`), `--frac` (default `0.15`) — mirrors
`compare_rotation_prob.py`'s `--prob` flag.

## Testing / validation

3 seeds first (`[42, 137, 271]`), full 300 generations, instance 100 clients
— the project's own fast-reject protocol. Report HV/GD/IGD/Spacing vs.
baseline (`frac=1.0`-equivalent, i.e. today's `run_qinsga3` unmodified) and
vs. the NSGA-III cache, plus chromosome diversity `Var(X_norm)` (the metric
this whole diagnostic chapter is built around) and per-run elapsed time (the
masking itself is O(pop_size × n_genes), no expected runtime regression).

If there's a positive signal (any metric with visible separation, ideally
significant or at the Mann-Whitney 3-seed floor p=0.10 like remedies F/G) →
scale to 20 seeds. If null → document alongside the 5 already-rejected
"tuning" attempts in `Solvers/IRP_results_summary.md`, recording the
diversity number either way so a null result stays informative.

## Non-goals

No change to production code. No test of alternate rotation gates (RQPSO/
PSO/chaotic) combined with partial masking. No adaptive/scheduled `frac` —
fixed and parametrised only, per this screening's scope.

## Result: stability criterion (3 seeds, 300 gen, instance 100)

`sensitivity/compare_partial_rotation.py`, `frac=0.15`:

| Indicateur | Baseline | Test (frac=0.15) | Mann-Whitney vs baseline |
|---|---|---|---|
| HV ↑ | 0.213276 | 0.172665 (-19%) | U=7.0, p=0.40 (non sig.) |
| GD ↓ | 0.480850 | 0.602452 (+25% worse) | U=1.0, p=0.20 (non sig.) |
| IGD ↓ | 0.543743 | 0.602943 (+11% worse) | U=2.0, p=0.40 (non sig.) |
| Diversité chromosome | 0.003984 | **0.020392 (×5.1)** | — |

The mechanism moved diversity by far more than any prior remedy
(`rotation_prob`: 0.003984 → 0.003727, essentially unchanged) — but quality
regressed on all three metrics, not significant at 3 seeds but directionally
consistent. Same "diversity up, quality down" pattern already seen for
Remèdes A and B — **not scaled to 20 seeds**, same decision rule as those two
("scale only on positive signal"). To be recorded in
`Solvers/IRP_results_summary.md` alongside the already-rejected "tuning"
attempts.

## Addendum: correlation-with-past-improvement criterion

The user's second proposed criterion ("genes most correlated with previous
improvements") has no existing per-gene tracking to reuse (unlike
`_elite_rms_distance` for the stability criterion) — this needed a new
mechanism, tested in a **separate** script (`sensitivity/
compare_partial_rotation_correlation.py`), same convention as every other
sibling remedy variant in `sensitivity/` (each hypothesis gets its own file).

**Why population-wide, not per-individual**: a per-individual history is
structurally broken by QINSGA3's elitist survival, which re-selects the
*whole* population from a freshly merged parent+offspring pool every
generation — the same "individual-identity mismatch" already documented in
`Solvers/QINSGA3/algorithm.py`'s module note above `_chaotic_lambda_step` for
`pbest`/`lambda` state. A per-gene, population-aggregate signal sidesteps
this entirely.

**Signal**: each generation, per gene `j`:
- `x_j(t)` = population-mean `|Δθ_j|` of the FULL (unmasked) rotation step —
  always computed, even for genes the mask would exclude from the real
  update, so a currently-excluded gene's score can still recover later (the
  lock-in a masked-only signal would otherwise cause).
- `y(t)` = `-mean(F_norm[pareto_idx])` (population quality proxy, shared
  across all genes).
- `corr_j` = rolling-window (`W=30` generations) Pearson correlation between
  the `x_j` and `y` histories, vectorised across genes.

**Warm-up**: first `W=30` generations rotate every gene fully (no history yet
to compute a correlation from). After warm-up, each generation selects the
`frac` (0.15) genes with the **highest** `corr_j` (gene's activity level
tracks quality improvement) to receive that generation's rotation update; the
rest keep their pre-rotation θ, same masking mechanism as the stability
variant, but with a mask that is **global to the population** (shape
`(n_genes,)`), not per-individual.

### Result (3 seeds, 300 gen, instance 100)

| Indicateur | Baseline | Test (frac=0.15, window=30) | Mann-Whitney vs baseline |
|---|---|---|---|
| HV ↑ | 0.213909 | 0.135927 (-36%) | **U=9.0, p=0.10 (total separation)** |
| GD ↓ | 0.476642 | 0.680655 (+43% worse) | **U=0.0, p=0.10 (total separation)** |
| IGD ↓ | 0.543711 | 0.648593 (+19% worse) | **U=0.0, p=0.10 (total separation)** |
| Diversité chromosome | 0.003984 | **0.011978 (×3.0)** | — |

Total separation at the 3-seed Mann-Whitney floor (p=0.10, same floor as
remèdes F/G's *positive* results) — but here every metric moves in the
**wrong** direction. Cleaner/stronger negative signal than the stability
criterion (which was directionally consistent but not at the separation
floor). Same "diversity up, quality down" conclusion — **not scaled to 20
seeds**. Both criteria now documented together as Remède H (rejected) in
`Solvers/IRP_results_summary.md`.
