# Model formulation (final)

Pharmaceutical many-objective inventory routing problem, reformulated after
the editorial desk-reject (IJITS Research) and an independent audit. This
document is the single technical reference. The CPLEX model (`models/`), the
metaheuristic decoder/evaluator (`solvers/nsga3/decoder.py`,
`solvers/nsga3/evaluator.py`), the local-search repair
(`solvers/qinsga3/repair.py`) and the independent checker
(`tests/solution_checker.py`) all implement exactly what is written here.
The same model in French, for the paper: `docs/modele_IRP_corrige.docx`.

## 1. Editor's remarks and how they are answered

| # | Remark | Answer |
|---|---|---|
| E1 | Replenishment in Sec. 3.2 (= quantity shipped) contradicts (c7) (= demand due) | The replenishment is now a **decision**: supplier order `r_p,t` with order indicator and fixed ordering cost (C6, C8). |
| E2 | Eq. (7) multiplied distance × load × a cost per km; empty return legs were free | Transport cost in three terms with their own units: €/route, €/vehicle-km (every arc, empty return included), €/(unit·km). Vehicles return empty (C4). |
| E3 | f4 varied only through the scaled depot inventory already in the holding cost | f4 = **peak net working capital** per period (stock value + receivables − payables), cyclic accounting and closed stock cycle (C7). A peak is not a scaled sum of stock. Payoff table in §9. |
| E4 | One synthetic instance per size | `data/replicates/` (independent instances per size). |
| E5 | ≈ 10.6 % more evaluations for the proposed method | Root cause removed (full re-evaluation after each migration and after the last generation). All algorithms run on the same budget **in evaluations** and report the exact count. |
| E6 | Different parallelisation; runtimes inconclusive | All algorithms evaluate serially by default, use the same final-front rule (feasible non-dominated set of the last population) and the same final repair. |
| E7 | Component analysis does not isolate the mechanisms | `scripts/run_ablation.py`: one mechanism off at a time, same budget and seeds, NSGA-III reference, paired Wilcoxon tests. |
| E8 | No representative plans, no independent formulation check | `scripts/export_plan.py` (knee solution: orders, stock, WC, routes); `tests/` (independent checker + CPLEX cross-check). |

## 2. Other changes made during the audit

**Model**
- Arrival and return times are per vehicle (they were shared by all vehicles, and all vehicles of a period had to return at the same instant).
- The old c14 (compatibility through `K_l,t`) is deleted: it forbade anticipating a cold delivery; C10 + C11 already enforce compatibility.
- No idle waiting: arrival times are exact on used arcs, as in the decoder (CPLEX could otherwise wait to avoid early penalties).
- f3 is the longest route of each period, summed (C19). The former total driving time was proportional to the kilometres like f2 (Pearson 0.98); now Pearson(f2, f3) = 0.20–0.30.
- Symmetry breaking applies only within groups of identical vehicles, per period (C17).
- The objective budgets (c15–c18, set to 99,999) are removed: inactive, and infeasible on large instances.
- The shift length τ^max is 10 h (was 20 h); fleets are also sized on time (70 % of the fleet's shift time covers the mandatory routes).
- Arc-specific big-M values; redundant constraints (return to depot, minimum load) removed; payoff table with a Pareto-optimal secondary step (other objectives minimised as a sum normalised by their ideals); reported objectives recomputed exactly after each solve.

**Algorithms**
- Delivery shift: right truck class, quantities ±δ. Inter-route moves keep the truck class and never visit a client twice; identical trucks are relabelled (C17).
- Every repair move must strictly improve f1 and not worsen f2, f3, f4.
- Decoder: opens the next truck before exceeding τ^max; orders never leave more stock than `I_init + remaining demand`, so the cycle can always close.
- Constraint vector: lateness per class; mutation rate 1/D counts all genes.

**Data and parameters** (strict loader: no silent defaults)
- Removed: `c_route_value`, `DIO`, `DSO`, `DPO`, `period_days`, `C_max`, `E_max`, `T_max`, `B`, `BIG_M`, `R_max_*`, `min_delivery_threshold`, `K_lt`, `tau_min`, `MOQ_*`.
- `h_O_space`, `alpha_r`, `e_stock` → `h_cold`, `h_ambient`; `requires_cold` (truck id) → boolean `cold`.

## 3. Sets

| Symbol | Meaning |
|---|---|
| `N = {0} ∪ C` | nodes; 0 = depot, C = clients |
| `A` | arcs (i, j), i ≠ j |
| `T = {1..n_T}` | periods (repeating cycle for the financial accounting) |
| `K = K^c ∪ K^a` | refrigerated / standard vehicles |
| `P = {c, a}` | product classes; `cl(l,τ)` = class of client l's demand due in τ. GDP ranges (2–8 °C, 15–25 °C) are disjoint: a truck carries one class |

## 4. Parameters (every key required in the instance JSON)

| Symbol | Key | Unit | Value in the instances |
|---|---|---|---|
| `q_l,τ`, `cl(l,τ)` | `q_lt`, `cold` | units, bool | data |
| `Q_k`, `v_k`, `K^c` | `Q`, `v`, `frigo_trucks` | units, km/h | data |
| coordinates → `d_ij` | `sets.coordinates` | km | data |
| `s` | `s_value` | h (client service = depot loading) | 0.25 |
| `τ^max` | `tau_max` | h | 10 |
| `ET_l`, `LT_l` | `ET`, `LT` (per client) | h (morning, afternoon or full-day reception window) | [0.5, 4.5], [3.5, 8] or [0.5, 8] |
| `c1`, `c2` | `c1`, `c2` | €/h early (arrival before the reception opens) / late (after it closes) | 20 / 40 |
| `F^veh` | `F_veh` | €/route | 20 |
| `c^km` | `c_km` | €/vehicle-km | 0.8 |
| `p5` | `p5` | €/h refrigeration (→ p5/v_k €/km) | 3 |
| `c^load` | `c_load` | €/(unit·km) | 0.002 |
| `I^init`, `I^min`, `I^max` | `I_O_init_*`, `I_O_min_*`, `I_O_max_*` | units | data |
| `h^c`, `h^a` | `h_cold`, `h_ambient` | €/(unit·period) | 0.305 / 0.3 |
| `F^ord` | `F_order` | €/order | 50 |
| `P^sale_l`, `P^pur` | `P_sale`, `P_purchase` | €/unit | data / 5 |
| `L^S`, `L^P` | `L_S`, `L_P` | periods (1 period = 15 days) | 3 (45 days) / 2 (30 days) |
| CMEM | `g, Cr, Cd, A_f, rho, a, w, kg_per_unit, e_co2, fuel_to_joules` | see §6 (fuel_to_joules in J/L) | 9.81, 0.01, 0.7, 3.912, 1.2041, 0, 6350, 10, 2.67, 36e6 (vehicle values of the PRP: Bektaş & Laporte 2011; Demir, Bektaş & Laporte 2012) |

Derived constants: time big-M `τ^max + s + d_ij/v_k` (≥ side) and `τ^max` (≤ side), and
`R̄_p = I^max_p − I^min_p + Σ_{k of class p} Q_k` (largest useful order, C8).

## 5. Decision variables

| Variable | Domain | Meaning |
|---|---|---|
| `x_ijtk` | {0,1} | vehicle k travels (i,j) in t |
| `f_ijtk` | ℤ≥0 | load on the arc (whole units) |
| `q'_l,t,τ` | ℤ≥0, t ≤ τ | delivered to l in t for the demand due in τ |
| `r_p,t`, `z_p,t` | ℤ≥0, {0,1} | supplier order of class p received in t; order placed |
| `I_p,t` | ≥ 0 | depot stock at the end of t |
| `a_i,t,k`, `e_t,k` | [0, τ^max] | arrival at i, return to the depot |
| `w1_ltk`, `w2_ltk` | ≥ 0 | earliness, lateness |
| `E_t` | [0, τ^max] | return time of the longest route of period t |
| `W` | free | peak net working capital |

Derived: `D_l,t = Σ_{τ≥t} q'_l,t,τ`; `u_l,t,k = Σ_i x_i,l,t,k`;
`win_L(t)` = periods t, t−1, …, t−L+1 taken cyclically.

## 6. Objectives (all minimised)

```
f1 = Σ_tk F^veh Σ_j x_0jtk + Σ_ijtk (c^km + [k∈K^c] p5/v_k) d_ij x_ijtk + Σ_ijtk c^load d_ij f_ijtk
   + Σ_t (h^c I_c,t + h^a I_a,t) + Σ_l,t,k (c1 w1_ltk + c2 w2_ltk) + F^ord Σ_p,t z_p,t              [€]

f2 = e_co2 / fuel_to_joules · Σ_ijtk d^m_ij [ (a + g Cr)(w x_ijtk + kg_per_unit f_ijtk) + ½ Cd A_f ρ v_k² x_ijtk ]   [kg]
     (linearised CMEM, Bektas & Laporte 2011; v_k in m/s, d^m in m)

f3 = Σ_t E_t,   E_t ≥ e_t,k   ∀k                                                                 (C19)   [h]

f4 = W,   W ≥ P^pur Σ_p I_p,t + Σ_l P^sale_l Σ_{t'∈win_{L^S}(t)} D_l,t' − P^pur Σ_p Σ_{t'∈win_{L^P}(t)} r_p,t'   ∀t   (C18)   [€]
```

## 7. Constraints

| # | Constraint |
|---|---|
| C1 | `Σ_j x_0jtk ≤ 1` |
| C2 | flow conservation at every client, per vehicle and period (implies the return to the depot); at most one visit per client, vehicle and period (implied by C13, stated explicitly) |
| C3 | `f_ijtk ≤ Q_k x_ijtk`, j ≠ 0 |
| C4 | `f_i0tk = 0` (return empty) |
| C5 | `I^min_p ≤ I_p,t ≤ I^max_p` |
| C6 | `I_p,t = I_p,t−1 + r_p,t − Σ_l Σ_{τ≥t, cl(l,τ)=p} q'_l,t,τ`, `I_p,0 = I^init_p` |
| C7 | `I_p,n_T = I^init_p` (closed cycle) |
| C8 | `r_p,t ≤ R̄_p z_p,t` |
| C9 | `Σ_{t≤τ} q'_l,t,τ = q_l,τ` (on time or early, never late) |
| C10 | `Σ_{k∈K^p} (Σ_{i∈N} f_iltk − Σ_{j∈N} f_ljtk) = Σ_{τ≥t, cl(l,τ)=p} q'_l,t,τ`   ∀l∈C, t∈T, p∈P |
| C11 | `Σ_{i∈N} f_iltk − Σ_{j∈N} f_ljtk ≥ u_l,t,k`  ∀l,t,k (no empty visit; with C3 and integer f, no subtour) and `Σ_{k∈K^p} u_l,t,k ≤ 1`  ∀l,t,p (one truck per class, client and period) |
| C12 | `a_0,t,k = 0` |
| C13 | `a_j,t,k = a_i,t,k + s + d_ij/v_k` if `x_ijtk = 1`, j ≠ 0 (big-M `τ^max + s + d_ij/v_k` on the ≥ side, `τ^max` on the ≤ side; for i = 0, s is the loading time) |
| C14 | `e_t,k = a_i,t,k + s + d_i0/v_k` if `x_i0tk = 1` |
| C15 | `e_t,k ≤ τ^max Σ_j x_0jtk` (shift length; e = 0 for an unused vehicle) |
| C16 | `w1_ltk ≥ ET_l − a_ltk − ET_l(1 − u_ltk)`, `w2_ltk ≥ a_ltk − LT_l − max(0, τ^max − LT_l)·(1 − u_ltk)` (soft windows: penalised in f1, never infeasible) |
| C17 | identical vehicles (class, Q, v), each period: `Σ_j x_0jtk ≥ Σ_j x_0jtk'` for k < k' |
| C18 | peak working capital (§6) |
| C19 | longest route of each period (§6) |

## 8. Metaheuristic encoding

Chromosome = `[q(l,t)] + [priority(l)] + [order(p,t) ∈ [0,1]]`, D = nC·nT + nC + 2·nT genes.

For each period, the decoder:
1. Requests deliveries from the quantity genes (mandatory floor = demand of this class still uncovered).
2. Builds routes truck after truck by nearest neighbour biased by the priority genes, opening the next truck before exceeding τ^max. Time windows are soft (penalised in f1 through c1, c2): the priority genes let evolution order the clients, e.g. morning windows before afternoon ones.
3. Sets the order: `r = r_min + g·(r_max − r_min)` with `r_min` keeping I ≥ I^min, `r_max` keeping I ≤ I^max and I ≤ I^init + remaining demand; in the last period the order closes the cycle exactly.

Search-space restrictions of the decoder (subsets of the MILP's feasible set):
- anticipation only in periods whose demand has the same class;
- optional deliveries below 5 units are dropped;
- nearest-neighbour route construction.

Constraint vector G (fixed length `nT + 2nC·nT + 4nT + 2`): shift length (C15), demand never late per class, i.e. never delivered after its due period (C9, C10), depot bounds (C5), closed cycle (C7). Time-window lateness (w2) is not in G: it is a soft cost in f1, exactly as in the MILP.

## 9. Verification and figures

`python -m pytest tests -q` (35 tests):
- independent checker vs decoder/evaluator on 6 instances, with and without repair, every repair move alone;
- CPLEX cross-check: decoded and repaired solutions fixed in the MILP are feasible there, with identical objectives;
- regression tests for every audit finding.

Payoff table, instance with 3 clients (CPLEX, all optimal). Each row: minimise the objective alone (its ideal), then keep it at that ideal and minimise the sum of the three other objectives, each divided by its own ideal value (not a lexicographic order f_a → f_b → …). Every row is therefore Pareto-optimal; rows f2 and f3 share f3 = 1.75 h but differ because the f3 row trades 0.07 kg of CO2 for 19.66 € of cost. On such a tiny instance the routes are very short (f3 = 1.75 h for three of the four rows); on instances of 100–200 clients, Pearson(f2, f3) is only 0.20–0.30:

| minimised | f1 (€) | f2 (kg) | f3 (h) | f4 (€) |
|---|---|---|---|---|
| f1 | **449.92** | 11.46 | 1.75 | 2,691.00 |
| f2 | 498.23 | **11.39** | 1.75 | 2,341.00 |
| f3 | 478.57 | 11.46 | **1.75** | 2,341.00 |
| f4 | 614.13 | 21.06 | 3.25 | **2,017.10** |

Over 300 random solutions (seed 0) of instances 100 and 200: Pearson(f2, f3) = 0.30 / 0.20; largest |Pearson| between any two objectives 0.45 / 0.62.

## 10. Experimental protocol

- Budget in evaluations `pop × n_gen` for all algorithms (pop 200; MOEA/D has 165 subproblems and runs more generations); exact count in `meta["n_evals"]`.
- Serial evaluation; same final-front rule; same final repair; same SBX/PM operators (p_c 0.9, η 20, p_m 1/D).
- 20 seeds, several instances per size, paired Wilcoxon tests.
- QI-NSGA-III starts θ at π/4 ± 0.05 by design (maximum superposition); report it.

## 11. Points to keep in mind for the paper

- The cost values of §4 (F_veh, c_km, c_load, F_ord, L_S, L_P) need a source or a justification.
- One period = 15 days with one delivery day (routes ≤ τ^max = 10 h); the payment terms L_S = 3 and L_P = 2 periods are 45 and 30 days.
- The horizon must satisfy n_T > max(L_S, L_P) (≥ 4 periods); otherwise the receivable window covers the whole horizon and receivables are constant. Instances with 3, 25 and 30 clients (n_T = 2–3) are for model validation with CPLEX only.
- Depot capacity is checked at the end of the period (cross-docking: the day's deliveries are prepared from the stock plus the day's arrival).
- Service time s applies at each client (unloading) and at the depot on departure (loading), not on return: the vehicle returns empty (C4).
- Time windows are per client (morning / afternoon / full day) and soft. With no idle waiting, an early arrival is not shifted but penalised: c1 is the cost of an early, unplanned arrival for the client (of the order of a driver's hourly cost), c2 = 2·c1.
- All results produced before this reformulation are obsolete. The old deliverables were removed; they remain in the git history.
