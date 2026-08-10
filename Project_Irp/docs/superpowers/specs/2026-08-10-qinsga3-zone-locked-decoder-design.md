# QINSGA3 zone-locked decoder — continuity screening design

## Context

`docs/superpowers/specs/2026-08-10-qinsga3-prins-split-decoder-design.md`
tested whether replacing greedy-fill with a DP-optimal split (Prins 2004)
of the same fixed giant-tour order would restore theta -> route continuity.
It did not — Prins-split was *worse* than both the original decoder and
the greedy-fill giant-tour prototype (Jaccard 0.32 vs 0.19 vs 0.08 at the
smallest production alpha). Its own recorded conclusion: DP optimality does
not imply stability — a global argmin over every contiguous partition is
just as exposed to near-ties as the original decoder's local, sequential
argmin. Every remedy tried so far (this session's giant-tour, incremental
decoder, damped priority, Prins-split; the eleven earlier remedies A-J in
`Solvers/IRP_results_summary.md`) changes *which decision rule* picks the
next client or the next partition, while leaving the full client set
equally reachable from anywhere in the order — a small perturbation can
still, in principle, move any client's assignment anywhere.

This design changes a different axis: not the decision rule, but the
*reach* of any single perturbation. Clients are partitioned into fixed
geographic zones, computed once from static coordinates, independent of
theta and never touched by the GA. Each zone is permanently mapped to one
truck (in the group's fixed truck order). A theta perturbation can still
reorder a zone's own clients (or drop one under capacity pressure), but it
can never move a client into a different truck's zone — bounding how far
any single cascade can reach, structurally, rather than trying to make the
decision rule itself smoother.

## Goal

Check whether locking clients to fixed geographic zones (one per truck)
produces a real theta -> route continuity gradient — the same
Jaccard-route-distance-vs-alpha test used for every prior remedy this
session — by construction bounding each truck's route to a fixed client
subset, so cascades within one zone can no longer contaminate another
truck's route.

## Design

### Coordinates (newly sourced — not currently exposed by `load_instance`)

`models/parametres.py::load_instance` reads `sets_raw["coordinates"]`
(`{client_id: [x, y]}`, present in `data/instance_100_clients.json`,
depot included at id `0`) only to build the distance dict `d` — it is
never returned in `sets_`/`params_`. This design's script reads the JSON
instance file directly (same path already available to every diagnostic
script) to get `{int(id): (x, y)}` without touching `load_instance`.

### Zoning (computed once per truck group, independent of theta)

Sweep partition (Gillett & Miller 1974): clients sorted by polar angle
around the depot's coordinates, then cut into contiguous angular sectors,
one per truck in the group's fixed order (`frigo_list` /
`non_frigo_trucks` — both static, from `params_`). Cut thresholds are
proportional to each truck's capacity `Q[truck]` against cumulative
**aggregate** demand (`sum(q_lt[l, t] for t in T)` per client, a static
per-client weight, not the smaller per-period `qty_dict` seen at any single
`build_routes` call) — keeps zones roughly capacity-balanced despite a
heterogeneous fleet. Zone membership is a client-level property: computed
once for the whole run, reused for every period and every generation.

```python
def _zone_assignment(coords, O, client_ids, trucks, Q, total_demand):
    ox, oy = coords[O]
    order = sorted(
        client_ids,
        key=lambda l: (math.atan2(coords[l][1] - oy, coords[l][0] - ox), l),
    )
    total_Q = sum(Q[truck] for truck in trucks) or 1.0
    total_d = sum(total_demand.get(l, 0.0) for l in client_ids)
    zone, cum, k_pos = {}, 0.0, 0
    cum_target = total_d * (Q[trucks[0]] / total_Q)
    for l in order:
        while k_pos < len(trucks) - 1 and cum >= cum_target:
            k_pos += 1
            cum_target += total_d * (Q[trucks[k_pos]] / total_Q)
        zone[l] = k_pos
        cum += total_demand.get(l, 0.0)
    return zone
```

### Construction (near-identical fork of `_nearest_neighbour`)

Reuses production's exact greedy loop and scoring
(`score = dist / (0.5 + priority)`, same mandatory-floor and tau_max-bypass
rules) — the only change is an added eligibility filter: truck at position
`truck_idx` only considers pending clients with `zone[l] == truck_idx`.
Priority (theta-driven) still freely reorders *within* a zone; it can never
move a client to a different zone/truck. Built as a closure over a
precomputed `{tuple(sorted(trucks)): zone_dict}` map (one entry per truck
group — frigo, non-frigo) so the resulting function keeps
`_nearest_neighbour`'s exact call signature and stays a drop-in
monkey-patch target.

**Known simplification (documented, not hidden — same convention as every
prior remedy)**: a client whose zone's truck can't fit it this period
(even at its mandatory floor) has no cross-zone fallback and stays
unserved this period, unlike production's free `truck_idx` roaming.
Tracked with a counter analogous to the giant-tour prototype's
`_skip_count`, reported by the smoke test, not verified non-degenerate
beforehand.

### Isolation

Same monkey-patch pattern as every remedy so far: `Solvers/NSGA3/decoder.py`
is never modified on disk.

## Scope

### Step 1 — continuity smoke test (decision gate)

`sensitivity/diagnose_zone_locked_continuity.py`, mirroring
`diagnose_giant_tour_continuity.py`/`diagnose_prins_split_continuity.py`
exactly: same real cached NSGA-III 100-client chromosomes, same sparse
alpha sweep, same Jaccard route-distance metric. Extends the comparison to
four columns (original / giant-tour greedy-fill / Prins-split /
zone-locked) for direct comparison against every decoder tried so far.

### Step 2 — quality campaign (conditional)

Only if step 1 shows a real gradient unlike the other three:
`sensitivity/compare_zone_locked.py`, same standard protocol as every
other remedy (3 seeds first, shared ideal/nadir, HV/GD/IGD/Spacing,
chromosome diversity, Mann-Whitney U vs. baseline and vs. NSGA-III).

If step 1 shows no real gradient, stop there and document the null result
alongside Prins-split and the other rejected remedies.

## Non-goals

No change to `Solvers/QINSGA3/algorithm.py` or `Solvers/NSGA3/decoder.py`
regardless of outcome — ablation-only in `sensitivity/`, same convention as
every prior remedy. No demand-weighted or capacity-rebalancing zoning
beyond the single proportional-cut rule above (e.g. no periodic
re-zoning, no dynamic rebalancing if a zone is consistently
over/under-loaded across periods). No cross-zone spillover for infeasible
clients (documented simplification above).

## Result: continuity smoke test (5 seeds, instance 100 clients)

`sensitivity/diagnose_zone_locked_continuity.py`, 78 pooled real NSGA-III
chromosomes, real production alpha schedule (`GENS_TO_TEST = [0, 50, 100,
150, 200, 250, 300]`):

| gen | alpha | Jaccard original | Jaccard giant-tour | Jaccard Prins-split | Jaccard zone-locked |
|---|---|---|---|---|---|
| 0 | 0.31416 | 0.7657 | 0.9355 | 0.9535 | 0.6062 |
| 50 | 0.26232 | 0.7224 | 0.9062 | 0.9282 | 0.5394 |
| 100 | 0.21049 | 0.6802 | 0.8866 | 0.9191 | 0.4969 |
| 150 | 0.15865 | 0.6727 | 0.9038 | 0.9315 | 0.4404 |
| 200 | 0.10681 | 0.5784 | 0.8460 | 0.8933 | 0.3500 |
| 250 | 0.05498 | 0.4293 | 0.7653 | 0.8383 | 0.2184 |
| 300 | 0.00314 | 0.0800 | 0.1881 | 0.3181 | 0.0217 |

Correlation (alpha, Jaccard): original r=0.8849, giant-tour r=0.7464,
Prins-split r=0.7178, **zone-locked r=0.9624**.

**Positive result — both interpretation criteria met, first time in this
whole remedy sequence (this session's giant-tour, Prins-split, and the
eleven earlier A-J remedies in `Solvers/IRP_results_summary.md`).** At the
smallest production alpha (end of run), zone-locked's Jaccard
route-distance (0.0217) is roughly **1/4 of the original decoder's**
(0.0800), and far below giant-tour (0.1881) and Prins-split (0.3181).
Zone-locked's correlation with alpha (r=0.9624) is also the *strongest* of
the four — a cleaner, more linear gradient, not the early saturation the
other three decoders show near the production alpha floor.

**Interpretation**: bounding the *reach* of a perturbation — rather than
changing the decision rule that resolves it — is the first mechanism in
this entire diagnostic campaign to move the needle on continuity itself.
Locking each client to a fixed geographic zone means a theta-driven
priority swing can still reorder or drop clients within its own zone (that
residual chaos is why zone-locked's Jaccard isn't zero), but it can never
relocate a client into a different truck's route the way every previous
decoder allowed — cutting off the long-range cascades that the original
diagnostic chapter's "decision margin" analysis identified as the
mechanism behind decoder chaos.

**Decision**: scale to a full quality campaign
(`sensitivity/compare_zone_locked.py`, Task 2 of the Scope section above)
— this is a follow-up plan of its own, not part of this one. Continuity
alone does not guarantee HV/GD/IGD gains (a zone-locked decoder also loses
the free cross-zone truck reassignment production relies on for capacity
slack, and the "known simplification" — no cross-zone spillover — could
cost served demand under some instances/periods; the smoke test's
`_unserved_count` was not inspected here and should be checked first in
the campaign script). But this is the first remedy in the whole sequence
worth carrying to that next stage on its own continuity signal.

## Result: quality campaign (3 seeds, 300 generations, instance 100 clients)

`sensitivity/compare_zone_locked.py`, standard protocol (shared ideal/nadir
with the NSGA-III cache, `repair_final_front=True` on both arms,
Mann-Whitney U). Fast smoke run (`--gen 5 --seeds 42`) confirmed the
pipeline runs end-to-end before committing to the full campaign — but
already surfaced 12 unserved clients at the final-front repair step alone,
an early warning that materialised at full scale:

| Indicateur | NSGA-III | Baseline (décodeur original) | Zone-locked (test) |
|---|---|---|---|
| HV ↑ | 0.532343 | 0.236667 | **0.001367** |
| GD ↓ | 0.168827 | 0.400671 | **1.124735** |
| IGD ↓ | 0.370646 | 0.478762 | **1.331274** |
| Spacing ↓ | 0.048527 | 0.034096 | 0.000000 |
| Front size | — | 26–62 | **1 (all 3 seeds)** |
| Temps moyen | — | 214.8s | 359.8s (~1.7x plus lent) |

Mann-Whitney U: baseline vs. zone-locked — total separation on every
metric (HV/GD/IGD U=9.0/0.0/0.0, Spacing U=9.0), p=0.100 (the 3-seed
Mann-Whitney floor, same structural limit already seen for remèdes F/G —
not an absence of effect, every single zone-locked seed was worse than
every single baseline seed). Zone-locked vs. NSGA-III: significant on all
four metrics (p<0.05, HV/GD/IGD at U=0.0/60.0/60.0).

Unserved-client count at the final-front repair step alone (main process
only — see the code comment in `compare_zone_locked.py::run_comparison`;
the actual 300-generation search's own unserved-client rate, evaluated
inside `ProcessPoolExecutor` workers, isn't observable through this
counter and is almost certainly higher): **23**, up from 12 at `gen=5` —
confirmed non-zero and growing, not a fluke of the short smoke run.

**Negative result — a genuinely different failure mode from every prior
remedy.** Every previous remedy (giant-tour, Prins-split, and the eleven
A-J remedies) failed by *not* restoring continuity. Zone-locked is the
first to actually restore it (see the smoke-test Result above) and *still*
collapse in quality — to a single-solution front on all 3 seeds, a level
of degeneracy well past ordinary "worse HV." **Root cause**: the "no
cross-zone spillover" simplification (documented in Design as a known
approximation, not verified non-degenerate beforehand) is far costlier on
this real instance than anticipated. Locking a client to one truck's zone
means a demand spike that truck's capacity can't absorb has no fallback —
unlike production's free `truck_idx` roaming, which lets any later truck
pick up the slack. This chronically violates the per-client cumulative-
demand constraint (`IRPProblemZoneLocked`'s G-constraint block, `cum_dem -
cum_del`, one of `2*|T|` + `|clients|*|T|` + `2*|T|` inequality
constraints), especially on the mandatory final period — `_penalised_F`
then penalises nearly the entire population so heavily that
`ReferenceDirectionSurvival` has almost nothing feasible left to select
from, collapsing the returned front to a single point.

**Decision**: not scaled further (same rule as every other remedy: scale
only on a positive signal). The continuity smoke test's positive result
stands on its own terms — bounding perturbation reach *is* the mechanism
that restores theta→route continuity, the first in this whole campaign to
do so — but this specific zone-locked implementation is disqualified for
production consideration by its quality collapse, independently
confirmed by both the quality drop (worse than NSGA-III, not just
baseline) and the added runtime cost (~1.7x slower, the opposite of
remedy G's `repair_final_front` variant, which was fast *and* effective).
**Candidate follow-up, not pursued in this campaign**: bounded cross-zone
spillover (e.g. overflow to the angularly-adjacent zone only, rather than
no fallback at all) might recover most of the continuity gain while
restoring feasibility — a genuinely different design from a straightforward
parameter tweak, since it changes which simplification the decoder makes,
not a magnitude/frequency dial. Documented here as a starting point if this
thread is picked back up, not attempted.
