# Résultats de validation — DTLZ (QINSGA3, IGD)

Protocole : 30 exécutions indépendantes par problème/dimension (mêmes 30 seeds que la validation NSGA-III classique), moteur `Validation/Benchmarking/algorithms/qinsga3/runner.py` + `core.py`.
Population, générations et directions de référence identiques au protocole Cui et al. (2025) déjà utilisé pour NSGA-III (paramètres tirés de l'article, non modifiés).

**Paramètres de l'article conservés tels quels** : N, G, directions de référence (p, H), et p_cross=1.0, η_cross=20, p_mut=1/D (alignés sur NSGA-III).

**Étape 5 — algorithme corrigé (sélection élitiste + variation en espace X + mutation d'échappement)** : depuis cette campagne, le moteur QINSGA3 partagé (`Solvers/QINSGA3/algorithm.py`) fusionne parents et enfants et ne garde que les meilleurs `pop_size` via `ReferenceDirectionSurvival` (même sélection environnementale que NSGA-III) à chaque génération, et le croisement (SBX)/mutation (PM) se font en espace X avec les opérateurs de pymoo (eta=20) au lieu de l'espace θ. Une mutation d'échappement (`escape_prob = (2/n_var)×0.15`, appliquée uniformément à tous les problèmes — pas de cas particulier par problème) réinitialise un petit nombre de gènes à un tirage aléatoire après l'étape X pour compenser la perte de l'ancienne mutation « forte » (porte NOT quantique). Voir `Solvers/QINSGA3/README.md` pour la justification complète.

**Étape 6 — normalisation partagée avec `survival.norm`** : la sélection des guides (qui pilote la rotation) recalculait auparavant son propre ideal/nadir à partir de zéro à chaque génération, alors que l'étape de survie élitiste utilisait déjà l'ideal/nadir stable et monotone que pymoo maintient en interne (`ReferenceDirectionSurvival.norm`) — deux estimations différentes du même concept dans la même génération. Cette campagne reflète le correctif : la sélection des guides réutilise désormais `survival.norm.ideal_point`/`nadir_point` (repli sur l'ancien calcul seulement à la génération 0, avant que `survival.do()` ait tourné une fois). Validé au préalable sur l'IRP réel (20 seeds, GD/IGD significatifs p<0.05, HV à la limite p=0.060 — voir `Solvers/IRP_results_summary.md`). L'appel `survival._do()` a aussi été remplacé par `survival.do()` (le point d'entrée public de pymoo) dans `Validation/Benchmarking/algorithms/qinsga3/core.py`, pour rester un miroir fidèle de `Solvers/QINSGA3/algorithm.py` — sans effet numérique attendu ici puisque ces problèmes DTLZ/MaF sont non contraints (`problem.has_constraints()` est faux, donc `do()` et `_do()` empruntent le même chemin de code chez pymoo).

**Paramètres propres à l'algorithme, réglés pour la convergence (hérités des campagnes précédentes, non ré-ablatés depuis l'étape 5 sauf mention contraire)** :
- `noise_scale=0` (au lieu de 0.02, réglage IRP) — le bruit de mesure sert uniquement à éviter l'effondrement des chromosomes IRP après arrondi entier ; inutile sur variables continues.
- `migration_period=5, n_migrate=20` (au lieu de 10/10, réglage IRP) — injection plus fréquente et plus large de solutions de l'archive externe.
- Restent au réglage IRP : α_max=0.10π, α_min=0.001π, rotation="tanh".

M3 = 3 objectifs, M4 = 4 objectifs. DTLZ5–DTLZ7 ne sont validés qu'en M3 (pas de front de référence pymoo au-delà).

---

## DTLZ1

### M3
| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.027533 | 0.032347 | 0.060678 | 0.035624 | 0.009070 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 0.029976 |
| 2 | 137 | 0.031323 |
| 3 | 271 | 0.043238 |
| 4 | 491 | 0.056758 |
| 5 | 613 | 0.028414 |
| 6 | 733 | 0.036564 |
| 7 | 857 | 0.033823 |
| 8 | 977 | 0.057649 |
| 9 | 1009 | 0.036094 |
| 10 | 1123 | 0.029961 |
| 11 | 1249 | 0.030199 |
| 12 | 1373 | 0.035117 |
| 13 | 1499 | 0.032553 |
| 14 | 1609 | 0.027706 |
| 15 | 1733 | 0.032882 |
| 16 | 1871 | 0.030812 |
| 17 | 1997 | 0.028425 |
| 18 | 2113 | 0.032141 |
| 19 | 2237 | 0.029062 |
| 20 | 2351 | 0.030315 |
| 21 | 2467 | 0.040858 |
| 22 | 2593 | 0.027533 |
| 23 | 2711 | 0.031604 |
| 24 | 2837 | 0.050936 |
| 25 | 2953 | 0.060678 |
| 26 | 3079 | 0.028345 |
| 27 | 3191 | 0.033367 |
| 28 | 3313 | 0.033805 |
| 29 | 3433 | 0.030088 |
| 30 | 3557 | 0.038510 |

### M4
| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.055618 | 0.067584 | 0.095714 | 0.069680 | 0.008502 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 0.069372 |
| 2 | 137 | 0.088827 |
| 3 | 271 | 0.095714 |
| 4 | 491 | 0.064449 |
| 5 | 613 | 0.065997 |
| 6 | 733 | 0.063301 |
| 7 | 857 | 0.065709 |
| 8 | 977 | 0.074048 |
| 9 | 1009 | 0.063715 |
| 10 | 1123 | 0.063192 |
| 11 | 1249 | 0.062495 |
| 12 | 1373 | 0.072668 |
| 13 | 1499 | 0.064982 |
| 14 | 1609 | 0.075389 |
| 15 | 1733 | 0.073429 |
| 16 | 1871 | 0.072487 |
| 17 | 1997 | 0.074522 |
| 18 | 2113 | 0.057154 |
| 19 | 2237 | 0.069172 |
| 20 | 2351 | 0.075842 |
| 21 | 2467 | 0.055618 |
| 22 | 2593 | 0.070550 |
| 23 | 2711 | 0.080376 |
| 24 | 2837 | 0.062858 |
| 25 | 2953 | 0.078676 |
| 26 | 3079 | 0.065743 |
| 27 | 3191 | 0.065858 |
| 28 | 3313 | 0.062134 |
| 29 | 3433 | 0.072612 |
| 30 | 3557 | 0.063501 |

---

## DTLZ2

### M3
| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.065895 | 0.071677 | 0.081608 | 0.072326 | 0.004049 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 0.073190 |
| 2 | 137 | 0.071153 |
| 3 | 271 | 0.072103 |
| 4 | 491 | 0.069316 |
| 5 | 613 | 0.074497 |
| 6 | 733 | 0.071734 |
| 7 | 857 | 0.069345 |
| 8 | 977 | 0.068395 |
| 9 | 1009 | 0.066253 |
| 10 | 1123 | 0.067363 |
| 11 | 1249 | 0.075836 |
| 12 | 1373 | 0.066625 |
| 13 | 1499 | 0.078033 |
| 14 | 1609 | 0.065895 |
| 15 | 1733 | 0.077509 |
| 16 | 1871 | 0.067560 |
| 17 | 1997 | 0.072523 |
| 18 | 2113 | 0.071993 |
| 19 | 2237 | 0.081608 |
| 20 | 2351 | 0.071102 |
| 21 | 2467 | 0.071529 |
| 22 | 2593 | 0.070994 |
| 23 | 2711 | 0.080486 |
| 24 | 2837 | 0.075968 |
| 25 | 2953 | 0.071619 |
| 26 | 3079 | 0.070970 |
| 27 | 3191 | 0.071816 |
| 28 | 3313 | 0.070634 |
| 29 | 3433 | 0.075327 |
| 30 | 3557 | 0.078396 |

### M4
| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.136593 | 0.147211 | 0.154437 | 0.147147 | 0.004781 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 0.145709 |
| 2 | 137 | 0.151177 |
| 3 | 271 | 0.154237 |
| 4 | 491 | 0.141521 |
| 5 | 613 | 0.140674 |
| 6 | 733 | 0.143368 |
| 7 | 857 | 0.146324 |
| 8 | 977 | 0.147142 |
| 9 | 1009 | 0.154394 |
| 10 | 1123 | 0.150739 |
| 11 | 1249 | 0.145178 |
| 12 | 1373 | 0.153037 |
| 13 | 1499 | 0.151697 |
| 14 | 1609 | 0.148779 |
| 15 | 1733 | 0.151130 |
| 16 | 1871 | 0.143133 |
| 17 | 1997 | 0.143957 |
| 18 | 2113 | 0.145157 |
| 19 | 2237 | 0.140032 |
| 20 | 2351 | 0.149203 |
| 21 | 2467 | 0.146550 |
| 22 | 2593 | 0.137987 |
| 23 | 2711 | 0.151226 |
| 24 | 2837 | 0.144056 |
| 25 | 2953 | 0.154437 |
| 26 | 3079 | 0.148621 |
| 27 | 3191 | 0.150697 |
| 28 | 3313 | 0.136593 |
| 29 | 3433 | 0.147280 |
| 30 | 3557 | 0.150386 |

---

## DTLZ3

### M3
| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.072882 | 0.122813 | 1.068804 | 0.181430 | 0.188917 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 0.115763 |
| 2 | 137 | 0.102573 |
| 3 | 271 | 0.212290 |
| 4 | 491 | 0.102856 |
| 5 | 613 | 0.205430 |
| 6 | 733 | 0.092386 |
| 7 | 857 | 0.155571 |
| 8 | 977 | 0.127358 |
| 9 | 1009 | 0.097583 |
| 10 | 1123 | 0.156680 |
| 11 | 1249 | 0.113662 |
| 12 | 1373 | 0.118485 |
| 13 | 1499 | 0.102898 |
| 14 | 1609 | 0.149173 |
| 15 | 1733 | 0.126911 |
| 16 | 1871 | 0.072882 |
| 17 | 1997 | 0.118716 |
| 18 | 2113 | 0.097101 |
| 19 | 2237 | 0.560235 |
| 20 | 2351 | 0.149694 |
| 21 | 2467 | 0.144024 |
| 22 | 2593 | 0.137479 |
| 23 | 2711 | 0.101602 |
| 24 | 2837 | 0.117583 |
| 25 | 2953 | 0.092292 |
| 26 | 3079 | 0.131036 |
| 27 | 3191 | 0.102394 |
| 28 | 3313 | 0.221679 |
| 29 | 3433 | 1.068804 |
| 30 | 3557 | 0.347772 |

### M4
| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.192736 | 0.261414 | 1.393011 | 0.322345 | 0.217214 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 0.244015 |
| 2 | 137 | 0.267158 |
| 3 | 271 | 0.277943 |
| 4 | 491 | 0.586432 |
| 5 | 613 | 0.252685 |
| 6 | 733 | 0.383512 |
| 7 | 857 | 0.208846 |
| 8 | 977 | 0.373091 |
| 9 | 1009 | 0.291392 |
| 10 | 1123 | 0.212758 |
| 11 | 1249 | 0.219602 |
| 12 | 1373 | 0.192736 |
| 13 | 1499 | 0.296749 |
| 14 | 1609 | 0.365181 |
| 15 | 1733 | 0.306438 |
| 16 | 1871 | 0.235079 |
| 17 | 1997 | 1.393011 |
| 18 | 2113 | 0.323189 |
| 19 | 2237 | 0.204924 |
| 20 | 2351 | 0.255670 |
| 21 | 2467 | 0.200997 |
| 22 | 2593 | 0.338289 |
| 23 | 2711 | 0.500209 |
| 24 | 2837 | 0.205922 |
| 25 | 2953 | 0.304889 |
| 26 | 3079 | 0.254972 |
| 27 | 3191 | 0.306715 |
| 28 | 3313 | 0.211578 |
| 29 | 3433 | 0.230567 |
| 30 | 3557 | 0.225796 |

---

## DTLZ4

### M3
| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.065729 | 0.072636 | 0.082466 | 0.073323 | 0.004551 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 0.069703 |
| 2 | 137 | 0.074530 |
| 3 | 271 | 0.070912 |
| 4 | 491 | 0.075407 |
| 5 | 613 | 0.069026 |
| 6 | 733 | 0.079712 |
| 7 | 857 | 0.069107 |
| 8 | 977 | 0.082466 |
| 9 | 1009 | 0.069419 |
| 10 | 1123 | 0.074886 |
| 11 | 1249 | 0.079478 |
| 12 | 1373 | 0.070856 |
| 13 | 1499 | 0.065729 |
| 14 | 1609 | 0.068127 |
| 15 | 1733 | 0.080384 |
| 16 | 1871 | 0.069642 |
| 17 | 1997 | 0.081444 |
| 18 | 2113 | 0.070527 |
| 19 | 2237 | 0.067361 |
| 20 | 2351 | 0.079444 |
| 21 | 2467 | 0.079714 |
| 22 | 2593 | 0.074479 |
| 23 | 2711 | 0.074442 |
| 24 | 2837 | 0.069035 |
| 25 | 2953 | 0.072672 |
| 26 | 3079 | 0.072600 |
| 27 | 3191 | 0.074059 |
| 28 | 3313 | 0.073538 |
| 29 | 3433 | 0.070638 |
| 30 | 3557 | 0.070356 |

### M4
| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.147725 | 0.156125 | 0.167251 | 0.156704 | 0.004975 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 0.163719 |
| 2 | 137 | 0.147725 |
| 3 | 271 | 0.158748 |
| 4 | 491 | 0.151372 |
| 5 | 613 | 0.150310 |
| 6 | 733 | 0.156902 |
| 7 | 857 | 0.163977 |
| 8 | 977 | 0.161639 |
| 9 | 1009 | 0.160730 |
| 10 | 1123 | 0.156054 |
| 11 | 1249 | 0.163112 |
| 12 | 1373 | 0.163131 |
| 13 | 1499 | 0.151285 |
| 14 | 1609 | 0.156433 |
| 15 | 1733 | 0.160438 |
| 16 | 1871 | 0.156117 |
| 17 | 1997 | 0.153944 |
| 18 | 2113 | 0.153499 |
| 19 | 2237 | 0.164432 |
| 20 | 2351 | 0.156806 |
| 21 | 2467 | 0.149808 |
| 22 | 2593 | 0.150985 |
| 23 | 2711 | 0.151283 |
| 24 | 2837 | 0.157141 |
| 25 | 2953 | 0.154142 |
| 26 | 3079 | 0.167251 |
| 27 | 3191 | 0.156133 |
| 28 | 3313 | 0.155520 |
| 29 | 3433 | 0.154071 |
| 30 | 3557 | 0.154403 |

---

## DTLZ5

| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.007557 | 0.008997 | 0.013561 | 0.009401 | 0.001268 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 0.009290 |
| 2 | 137 | 0.011007 |
| 3 | 271 | 0.008626 |
| 4 | 491 | 0.009386 |
| 5 | 613 | 0.007941 |
| 6 | 733 | 0.009328 |
| 7 | 857 | 0.010116 |
| 8 | 977 | 0.009411 |
| 9 | 1009 | 0.007557 |
| 10 | 1123 | 0.008591 |
| 11 | 1249 | 0.009994 |
| 12 | 1373 | 0.008834 |
| 13 | 1499 | 0.008956 |
| 14 | 1609 | 0.008285 |
| 15 | 1733 | 0.011773 |
| 16 | 1871 | 0.009038 |
| 17 | 1997 | 0.008918 |
| 18 | 2113 | 0.008617 |
| 19 | 2237 | 0.008781 |
| 20 | 2351 | 0.008042 |
| 21 | 2467 | 0.009516 |
| 22 | 2593 | 0.010422 |
| 23 | 2711 | 0.008654 |
| 24 | 2837 | 0.013561 |
| 25 | 2953 | 0.010241 |
| 26 | 3079 | 0.009789 |
| 27 | 3191 | 0.008540 |
| 28 | 3313 | 0.008446 |
| 29 | 3433 | 0.008641 |
| 30 | 3557 | 0.011721 |

---

## DTLZ6

| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 1.564528 | 1.840963 | 2.171592 | 1.841740 | 0.125525 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 1.897170 |
| 2 | 137 | 1.827627 |
| 3 | 271 | 1.875708 |
| 4 | 491 | 2.013002 |
| 5 | 613 | 1.881826 |
| 6 | 733 | 1.737230 |
| 7 | 857 | 1.809704 |
| 8 | 977 | 1.996190 |
| 9 | 1009 | 1.835598 |
| 10 | 1123 | 1.955780 |
| 11 | 1249 | 2.171592 |
| 12 | 1373 | 1.844347 |
| 13 | 1499 | 1.564528 |
| 14 | 1609 | 1.900177 |
| 15 | 1733 | 1.879975 |
| 16 | 1871 | 1.786308 |
| 17 | 1997 | 1.837580 |
| 18 | 2113 | 2.014554 |
| 19 | 2237 | 1.823234 |
| 20 | 2351 | 1.616654 |
| 21 | 2467 | 1.787328 |
| 22 | 2593 | 1.887745 |
| 23 | 2711 | 1.688827 |
| 24 | 2837 | 1.591260 |
| 25 | 2953 | 1.819658 |
| 26 | 3079 | 1.912541 |
| 27 | 3191 | 1.774223 |
| 28 | 3313 | 1.748510 |
| 29 | 3433 | 1.887651 |
| 30 | 3557 | 1.885680 |

---

## DTLZ7

| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.092715 | 0.105321 | 0.125244 | 0.107515 | 0.007799 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 0.104676 |
| 2 | 137 | 0.106277 |
| 3 | 271 | 0.102140 |
| 4 | 491 | 0.100702 |
| 5 | 613 | 0.101335 |
| 6 | 733 | 0.100530 |
| 7 | 857 | 0.120786 |
| 8 | 977 | 0.114226 |
| 9 | 1009 | 0.125126 |
| 10 | 1123 | 0.092715 |
| 11 | 1249 | 0.103291 |
| 12 | 1373 | 0.102965 |
| 13 | 1499 | 0.105966 |
| 14 | 1609 | 0.106329 |
| 15 | 1733 | 0.102322 |
| 16 | 1871 | 0.113789 |
| 17 | 1997 | 0.106402 |
| 18 | 2113 | 0.114248 |
| 19 | 2237 | 0.110596 |
| 20 | 2351 | 0.103482 |
| 21 | 2467 | 0.116020 |
| 22 | 2593 | 0.103242 |
| 23 | 2711 | 0.101582 |
| 24 | 2837 | 0.103340 |
| 25 | 2953 | 0.096873 |
| 26 | 3079 | 0.102710 |
| 27 | 3191 | 0.114843 |
| 28 | 3313 | 0.125244 |
| 29 | 3433 | 0.116123 |
| 30 | 3557 | 0.107562 |

---

## Récapitulatif global

### M3

| Problem | IGD_Best | IGD_Median | IGD_Worst | IGD_Mean | IGD_Std |
|---|---|---|---|---|---|
| DTLZ1 | 0.027533 | 0.032347 | 0.060678 | 0.035624 | 0.009070 |
| DTLZ2 | 0.065895 | 0.071677 | 0.081608 | 0.072326 | 0.004049 |
| DTLZ3 | 0.072882 | 0.122813 | 1.068804 | 0.181430 | 0.188917 |
| DTLZ4 | 0.065729 | 0.072636 | 0.082466 | 0.073323 | 0.004551 |
| DTLZ5 | 0.007557 | 0.008997 | 0.013561 | 0.009401 | 0.001268 |
| DTLZ6 | 1.564528 | 1.840963 | 2.171592 | 1.841740 | 0.125525 |
| DTLZ7 | 0.092715 | 0.105321 | 0.125244 | 0.107515 | 0.007799 |

### M4

| Problem | IGD_Best | IGD_Median | IGD_Worst | IGD_Mean | IGD_Std |
|---|---|---|---|---|---|
| DTLZ1 | 0.055618 | 0.067584 | 0.095714 | 0.069680 | 0.008502 |
| DTLZ2 | 0.136593 | 0.147211 | 0.154437 | 0.147147 | 0.004781 |
| DTLZ3 | 0.192736 | 0.261414 | 1.393011 | 0.322345 | 0.217214 |
| DTLZ4 | 0.147725 | 0.156125 | 0.167251 | 0.156704 | 0.004975 |

---

## Comparaison avec NSGA-III classique (Mean IGD)

| Problem | M | NSGA-III | QINSGA3 (étape 6, corrigé) | Verdict |
|---|---|---|---|---|
| DTLZ1 | 3 | 0.228815 | **0.035624** | QINSGA3 gagne |
| DTLZ2 | 3 | 0.054468 | **0.072326** | NSGA-III gagne |
| DTLZ3 | 3 | 7.853266 | **0.181430** | QINSGA3 gagne |
| DTLZ4 | 3 | 0.152405 | **0.073323** | QINSGA3 gagne |
| DTLZ5 | 3 | 0.037518 | **0.009401** | QINSGA3 gagne |
| DTLZ6 | 3 | 0.716894 | **1.841740** | NSGA-III gagne |
| DTLZ7 | 3 | 0.113257 | **0.107515** | QINSGA3 gagne |
| DTLZ1 | 4 | 0.214947 | **0.069680** | QINSGA3 gagne |
| DTLZ2 | 4 | 0.121128 | **0.147147** | NSGA-III gagne |
| DTLZ3 | 4 | 5.785954 | **0.322345** | QINSGA3 gagne |
| DTLZ4 | 4 | 0.142183 | **0.156704** | NSGA-III gagne |

**Correctif DTLZ4** : avant la mutation d'échappement, DTLZ4 s'effondrait totalement (mean IGD identique à 6 décimales sur les 30 seeds — 0.945922 en M3, 1.045499 en M4). Cause : `QuantumPopulation` démarre tous les individus à θ=π/4 (x≈0.5 pour tout le monde), et DTLZ4 élève ses variables de position à la puissance 100 — 0.5¹⁰⁰≈0, donc 2 des 3(4) objectifs collapsent à 0 dès la génération 0, indépendamment de la seed. La mutation d'échappement restaure une capacité d'évasion perdue lors du passage du croisement/mutation en espace X (l'ancienne mutation « forte » en θ le faisait par accident). Voir `Solvers/QINSGA3/README.md` et `Validation/Benchmarking/algorithms/qinsga3/runner.py` pour le détail complet.
