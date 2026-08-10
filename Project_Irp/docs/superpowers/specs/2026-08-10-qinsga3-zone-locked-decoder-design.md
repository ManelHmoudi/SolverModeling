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
