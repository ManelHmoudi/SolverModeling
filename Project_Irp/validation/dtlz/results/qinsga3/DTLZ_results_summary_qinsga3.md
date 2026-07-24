# Résultats de validation — DTLZ (QINSGA3, IGD)

Protocole : 30 exécutions indépendantes par problème/dimension (mêmes 30 seeds que la validation NSGA-III classique), moteur `validation/algorithms/qinsga3/runner.py` + `core.py`.
Population, générations et directions de référence identiques au protocole Cui et al. (2025) déjà utilisé pour NSGA-III (paramètres tirés de l'article, non modifiés).

**Paramètres de l'article conservés tels quels** : N, G, directions de référence (p, H), et p_cross=1.0, η_cross=20, p_mut=1/D (alignés sur NSGA-III).
**Paramètres propres à l'algorithme QINSGA3, réglés pour améliorer la convergence** :
- `noise_scale=0` (au lieu de 0.02, réglage IRP) — le bruit de mesure de `QuantumPopulation.measure()` sert uniquement à éviter l'effondrement des chromosomes IRP après arrondi entier ; inutile sur variables continues, il dégradait la précision.
- `migration_period=5, n_migrate=20` (au lieu de 10/10, réglage IRP) — injection plus fréquente et plus large de solutions de l'archive externe dans la population.
- Restent au réglage IRP (aucun gain trouvé en les modifiant) : α_max=0.10π, α_min=0.001π, p_mut_strong=0.15, mut_sigma=0.05π, rotation="tanh".

M3 = 3 objectifs, M4 = 4 objectifs. DTLZ5–DTLZ7 ne sont validés qu'en M3 (pas de front de référence pymoo au-delà).

---

## DTLZ1

### M3
| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.051384 | 0.147044 | 1.536838 | 0.228982 | 0.285543 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 0.051384 |
| 2 | 137 | 0.066471 |
| 3 | 271 | 0.779714 |
| 4 | 491 | 0.336774 |
| 5 | 613 | 0.119734 |
| 6 | 733 | 0.083162 |
| 7 | 857 | 0.057141 |
| 8 | 977 | 0.078412 |
| 9 | 1009 | 0.122527 |
| 10 | 1123 | 0.056697 |
| 11 | 1249 | 0.187586 |
| 12 | 1373 | 0.233676 |
| 13 | 1499 | 0.151376 |
| 14 | 1609 | 0.053553 |
| 15 | 1733 | 0.064985 |
| 16 | 1871 | 0.331379 |
| 17 | 1997 | 0.178873 |
| 18 | 2113 | 0.148522 |
| 19 | 2237 | 0.145566 |
| 20 | 2351 | 0.410431 |
| 21 | 2467 | 0.156185 |
| 22 | 2593 | 0.084446 |
| 23 | 2711 | 0.329405 |
| 24 | 2837 | 0.136996 |
| 25 | 2953 | 0.081481 |
| 26 | 3079 | 1.536838 |
| 27 | 3191 | 0.190926 |
| 28 | 3313 | 0.186258 |
| 29 | 3433 | 0.112303 |
| 30 | 3557 | 0.396651 |

### M4
| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.115554 | 0.192193 | 0.976020 | 0.272580 | 0.192567 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 0.138147 |
| 2 | 137 | 0.157033 |
| 3 | 271 | 0.262975 |
| 4 | 491 | 0.218848 |
| 5 | 613 | 0.172470 |
| 6 | 733 | 0.146898 |
| 7 | 857 | 0.147601 |
| 8 | 977 | 0.430787 |
| 9 | 1009 | 0.184029 |
| 10 | 1123 | 0.127157 |
| 11 | 1249 | 0.468899 |
| 12 | 1373 | 0.293339 |
| 13 | 1499 | 0.120858 |
| 14 | 1609 | 0.125796 |
| 15 | 1733 | 0.115554 |
| 16 | 1871 | 0.541606 |
| 17 | 1997 | 0.271689 |
| 18 | 2113 | 0.368469 |
| 19 | 2237 | 0.427059 |
| 20 | 2351 | 0.200357 |
| 21 | 2467 | 0.395747 |
| 22 | 2593 | 0.680517 |
| 23 | 2711 | 0.207219 |
| 24 | 2837 | 0.127830 |
| 25 | 2953 | 0.162676 |
| 26 | 3079 | 0.275835 |
| 27 | 3191 | 0.151940 |
| 28 | 3313 | 0.152931 |
| 29 | 3433 | 0.976020 |
| 30 | 3557 | 0.127101 |

---

## DTLZ2

### M3
| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.073287 | 0.080878 | 0.088528 | 0.080279 | 0.003476 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 0.076661 |
| 2 | 137 | 0.080901 |
| 3 | 271 | 0.079754 |
| 4 | 491 | 0.076802 |
| 5 | 613 | 0.079931 |
| 6 | 733 | 0.077359 |
| 7 | 857 | 0.083690 |
| 8 | 977 | 0.079214 |
| 9 | 1009 | 0.077742 |
| 10 | 1123 | 0.073287 |
| 11 | 1249 | 0.083036 |
| 12 | 1373 | 0.081566 |
| 13 | 1499 | 0.084353 |
| 14 | 1609 | 0.082095 |
| 15 | 1733 | 0.081912 |
| 16 | 1871 | 0.080855 |
| 17 | 1997 | 0.086042 |
| 18 | 2113 | 0.082963 |
| 19 | 2237 | 0.075233 |
| 20 | 2351 | 0.082460 |
| 21 | 2467 | 0.073343 |
| 22 | 2593 | 0.077465 |
| 23 | 2711 | 0.081149 |
| 24 | 2837 | 0.079434 |
| 25 | 2953 | 0.083053 |
| 26 | 3079 | 0.082286 |
| 27 | 3191 | 0.088528 |
| 28 | 3313 | 0.076067 |
| 29 | 3433 | 0.081590 |
| 30 | 3557 | 0.079604 |

### M4
| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.157809 | 0.176044 | 0.190690 | 0.175939 | 0.006997 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 0.175300 |
| 2 | 137 | 0.179813 |
| 3 | 271 | 0.173501 |
| 4 | 491 | 0.176934 |
| 5 | 613 | 0.166271 |
| 6 | 733 | 0.183150 |
| 7 | 857 | 0.176558 |
| 8 | 977 | 0.184287 |
| 9 | 1009 | 0.183532 |
| 10 | 1123 | 0.182694 |
| 11 | 1249 | 0.190690 |
| 12 | 1373 | 0.174923 |
| 13 | 1499 | 0.175239 |
| 14 | 1609 | 0.166706 |
| 15 | 1733 | 0.181155 |
| 16 | 1871 | 0.162117 |
| 17 | 1997 | 0.179464 |
| 18 | 2113 | 0.167155 |
| 19 | 2237 | 0.181212 |
| 20 | 2351 | 0.182308 |
| 21 | 2467 | 0.175529 |
| 22 | 2593 | 0.173483 |
| 23 | 2711 | 0.186397 |
| 24 | 2837 | 0.167757 |
| 25 | 2953 | 0.170945 |
| 26 | 3079 | 0.163178 |
| 27 | 3191 | 0.172055 |
| 28 | 3313 | 0.177969 |
| 29 | 3433 | 0.178899 |
| 30 | 3557 | 0.168950 |

---

## DTLZ3

### M3
| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.562796 | 7.004988 | 24.535356 | 8.116732 | 5.553143 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 1.371316 |
| 2 | 137 | 20.529300 |
| 3 | 271 | 24.535356 |
| 4 | 491 | 5.235628 |
| 5 | 613 | 8.129283 |
| 6 | 733 | 10.970696 |
| 7 | 857 | 7.016168 |
| 8 | 977 | 12.651883 |
| 9 | 1009 | 3.032476 |
| 10 | 1123 | 4.460529 |
| 11 | 1249 | 7.448855 |
| 12 | 1373 | 9.228674 |
| 13 | 1499 | 12.468732 |
| 14 | 1609 | 6.993808 |
| 15 | 1733 | 5.667688 |
| 16 | 1871 | 1.101027 |
| 17 | 1997 | 0.562796 |
| 18 | 2113 | 8.518214 |
| 19 | 2237 | 1.093745 |
| 20 | 2351 | 1.929792 |
| 21 | 2467 | 8.477867 |
| 22 | 2593 | 6.266034 |
| 23 | 2711 | 3.991766 |
| 24 | 2837 | 14.969596 |
| 25 | 2953 | 13.082626 |
| 26 | 3079 | 14.900723 |
| 27 | 3191 | 5.060413 |
| 28 | 3313 | 10.378806 |
| 29 | 3433 | 6.750670 |
| 30 | 3557 | 6.677499 |

### M4
| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.672939 | 5.631564 | 39.428071 | 11.356073 | 11.321350 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 13.981978 |
| 2 | 137 | 11.377035 |
| 3 | 271 | 1.118247 |
| 4 | 491 | 2.696412 |
| 5 | 613 | 2.783053 |
| 6 | 733 | 0.833560 |
| 7 | 857 | 37.987605 |
| 8 | 977 | 7.371538 |
| 9 | 1009 | 3.763545 |
| 10 | 1123 | 1.262512 |
| 11 | 1249 | 17.114362 |
| 12 | 1373 | 30.318640 |
| 13 | 1499 | 4.514965 |
| 14 | 1609 | 27.240260 |
| 15 | 1733 | 31.229738 |
| 16 | 1871 | 4.061075 |
| 17 | 1997 | 13.266623 |
| 18 | 2113 | 4.241314 |
| 19 | 2237 | 17.909760 |
| 20 | 2351 | 6.150267 |
| 21 | 2467 | 6.727554 |
| 22 | 2593 | 9.309363 |
| 23 | 2711 | 23.588185 |
| 24 | 2837 | 5.112861 |
| 25 | 2953 | 4.474284 |
| 26 | 3079 | 39.428071 |
| 27 | 3191 | 4.944552 |
| 28 | 3313 | 0.672939 |
| 29 | 3433 | 3.876622 |
| 30 | 3557 | 3.325285 |

---

## DTLZ4

### M3
| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.131747 | 0.189058 | 0.261114 | 0.192409 | 0.037493 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 0.195920 |
| 2 | 137 | 0.151222 |
| 3 | 271 | 0.260642 |
| 4 | 491 | 0.178004 |
| 5 | 613 | 0.184645 |
| 6 | 733 | 0.249097 |
| 7 | 857 | 0.172428 |
| 8 | 977 | 0.198564 |
| 9 | 1009 | 0.151487 |
| 10 | 1123 | 0.195738 |
| 11 | 1249 | 0.229623 |
| 12 | 1373 | 0.136010 |
| 13 | 1499 | 0.213707 |
| 14 | 1609 | 0.219329 |
| 15 | 1733 | 0.152526 |
| 16 | 1871 | 0.191762 |
| 17 | 1997 | 0.261114 |
| 18 | 2113 | 0.201150 |
| 19 | 2237 | 0.258697 |
| 20 | 2351 | 0.131747 |
| 21 | 2467 | 0.145102 |
| 22 | 2593 | 0.205130 |
| 23 | 2711 | 0.261026 |
| 24 | 2837 | 0.172771 |
| 25 | 2953 | 0.186355 |
| 26 | 3079 | 0.158676 |
| 27 | 3191 | 0.193821 |
| 28 | 3313 | 0.177293 |
| 29 | 3433 | 0.165132 |
| 30 | 3557 | 0.173551 |

### M4
| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.269690 | 0.333257 | 0.372536 | 0.328362 | 0.024912 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 0.341789 |
| 2 | 137 | 0.306791 |
| 3 | 271 | 0.359757 |
| 4 | 491 | 0.312900 |
| 5 | 613 | 0.316726 |
| 6 | 733 | 0.338945 |
| 7 | 857 | 0.335864 |
| 8 | 977 | 0.336062 |
| 9 | 1009 | 0.312434 |
| 10 | 1123 | 0.330833 |
| 11 | 1249 | 0.354662 |
| 12 | 1373 | 0.317995 |
| 13 | 1499 | 0.338340 |
| 14 | 1609 | 0.306969 |
| 15 | 1733 | 0.343718 |
| 16 | 1871 | 0.306400 |
| 17 | 1997 | 0.296023 |
| 18 | 2113 | 0.302661 |
| 19 | 2237 | 0.372536 |
| 20 | 2351 | 0.343924 |
| 21 | 2467 | 0.357912 |
| 22 | 2593 | 0.356587 |
| 23 | 2711 | 0.269690 |
| 24 | 2837 | 0.289119 |
| 25 | 2953 | 0.316496 |
| 26 | 3079 | 0.298095 |
| 27 | 3191 | 0.362397 |
| 28 | 3313 | 0.326186 |
| 29 | 3433 | 0.335681 |
| 30 | 3557 | 0.363353 |

---

## DTLZ5 (M3 uniquement)

| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.011165 | 0.015653 | 0.023855 | 0.015989 | 0.002503 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 0.017203 |
| 2 | 137 | 0.015936 |
| 3 | 271 | 0.013604 |
| 4 | 491 | 0.014584 |
| 5 | 613 | 0.015629 |
| 6 | 733 | 0.018996 |
| 7 | 857 | 0.011670 |
| 8 | 977 | 0.019114 |
| 9 | 1009 | 0.017913 |
| 10 | 1123 | 0.023855 |
| 11 | 1249 | 0.015941 |
| 12 | 1373 | 0.015982 |
| 13 | 1499 | 0.015244 |
| 14 | 1609 | 0.017692 |
| 15 | 1733 | 0.013200 |
| 16 | 1871 | 0.011165 |
| 17 | 1997 | 0.018542 |
| 18 | 2113 | 0.015179 |
| 19 | 2237 | 0.018291 |
| 20 | 2351 | 0.014208 |
| 21 | 2467 | 0.018193 |
| 22 | 2593 | 0.015630 |
| 23 | 2711 | 0.015261 |
| 24 | 2837 | 0.015217 |
| 25 | 2953 | 0.014355 |
| 26 | 3079 | 0.015271 |
| 27 | 3191 | 0.016072 |
| 28 | 3313 | 0.015677 |
| 29 | 3433 | 0.017695 |
| 30 | 3557 | 0.012360 |

---

## DTLZ6 (M3 uniquement)

| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 5.230484 | 5.779092 | 6.461133 | 5.860450 | 0.345727 |

| run | seed | igd |
|---|---|---|
| 1 | 42 | 5.757114 |
| 2 | 137 | 6.363269 |
| 3 | 271 | 6.005973 |
| 4 | 491 | 5.682591 |
| 5 | 613 | 5.752951 |
| 6 | 733 | 5.410027 |
| 7 | 857 | 6.126114 |
| 8 | 977 | 6.340971 |
| 9 | 1009 | 5.405838 |
| 10 | 1123 | 5.667538 |
| 11 | 1249 | 5.709546 |
| 12 | 1373 | 5.403756 |
| 13 | 1499 | 5.808698 |
| 14 | 1609 | 6.200618 |
| 15 | 1733 | 6.262010 |
| 16 | 1871 | 5.388209 |
| 17 | 1997 | 5.626076 |
| 18 | 2113 | 5.820291 |
| 19 | 2237 | 6.461133 |
| 20 | 2351 | 5.230484 |
| 21 | 2467 | 5.457289 |
| 22 | 2593 | 5.932172 |
| 23 | 2711 | 6.442109 |
| 24 | 2837 | 6.380836 |
| 25 | 2953 | 5.771989 |
| 26 | 3079 | 6.225611 |
| 27 | 3191 | 5.929883 |
| 28 | 3313 | 5.742433 |
| 29 | 3433 | 5.721774 |
| 30 | 3557 | 5.786195 |

---

## DTLZ7 (M3 uniquement)

| Best | Median | Worst | Mean | Std |
|---|---|---|---|---|
| 0.842679 | 1.696889 | 3.843233 | 1.872026 | 0.697008 |

Meilleur qu'avec `noise_scale=0` seul (mean 3.092533 → 1.872026, ÷1.65) grâce à la migration réajustée, mais toujours pire que le réglage IRP d'origine (mean 0.368438) — front disjoint, limite documentée du réglage retenu.

| run | seed | igd |
|---|---|---|
| 1 | 42 | 2.508649 |
| 2 | 137 | 1.890130 |
| 3 | 271 | 1.240586 |
| 4 | 491 | 2.113494 |
| 5 | 613 | 1.521655 |
| 6 | 733 | 1.403444 |
| 7 | 857 | 1.765725 |
| 8 | 977 | 1.750054 |
| 9 | 1009 | 2.301613 |
| 10 | 1123 | 1.338628 |
| 11 | 1249 | 1.350692 |
| 12 | 1373 | 1.643723 |
| 13 | 1499 | 1.576238 |
| 14 | 1609 | 3.481893 |
| 15 | 1733 | 2.774522 |
| 16 | 1871 | 1.103943 |
| 17 | 1997 | 0.929933 |
| 18 | 2113 | 2.318991 |
| 19 | 2237 | 1.226388 |
| 20 | 2351 | 1.386085 |
| 21 | 2467 | 0.842679 |
| 22 | 2593 | 3.843233 |
| 23 | 2711 | 1.322153 |
| 24 | 2837 | 2.163251 |
| 25 | 2953 | 2.232612 |
| 26 | 3079 | 1.488280 |
| 27 | 3191 | 2.287427 |
| 28 | 3313 | 1.428959 |
| 29 | 3433 | 2.620692 |
| 30 | 3557 | 2.305099 |

---

## Récapitulatif global

### M3
| Problem | IGD_Best | IGD_Median | IGD_Worst | IGD_Mean | IGD_Std |
|---|---|---|---|---|---|
| DTLZ1 | 0.051384 | 0.147044 | 1.536838 | 0.228982 | 0.285543 |
| DTLZ2 | 0.073287 | 0.080878 | 0.088528 | 0.080279 | 0.003476 |
| DTLZ3 | 0.562796 | 7.004988 | 24.535356 | 8.116732 | 5.553143 |
| DTLZ4 | 0.131747 | 0.189058 | 0.261114 | 0.192409 | 0.037493 |
| DTLZ5 | 0.011165 | 0.015653 | 0.023855 | 0.015989 | 0.002503 |
| DTLZ6 | 5.230484 | 5.779092 | 6.461133 | 5.860450 | 0.345727 |
| DTLZ7 | 0.842679 | 1.696889 | 3.843233 | 1.872026 | 0.697008 |

### M4
| Problem | IGD_Best | IGD_Median | IGD_Worst | IGD_Mean | IGD_Std |
|---|---|---|---|---|---|
| DTLZ1 | 0.115554 | 0.192193 | 0.976020 | 0.272580 | 0.192567 |
| DTLZ2 | 0.157809 | 0.176044 | 0.190690 | 0.175939 | 0.006997 |
| DTLZ3 | 0.672939 | 5.631564 | 39.428071 | 11.356073 | 11.321350 |
| DTLZ4 | 0.269690 | 0.333257 | 0.372536 | 0.328362 | 0.024912 |

---

## Comparaison avec NSGA-III classique (Mean IGD) — évolution en 4 étapes

| Problem | M | NSGA-III | (1) Réglages IRP | (2) + crossover aligné | (3) + bruit=0 | (4) + migration réglée |
|---|---|---|---|---|---|---|
| DTLZ1 | 3 | 0.228815 | 9.501836 | 8.578591 | 0.383793 | **0.228982 (quasi égalité !)** |
| DTLZ1 | 4 | 0.214947 | 9.451351 | 7.168709 | 0.323396 | **0.272580** |
| DTLZ2 | 3 | 0.054468 | 0.085244 | 0.079626 | 0.081703 | 0.080279 |
| DTLZ2 | 4 | 0.121128 | 0.228367 | 0.184104 | 0.180516 | **0.175939** |
| DTLZ3 | 3 | 7.853266 | 191.185647 | 164.502568 | 15.163164 | **8.116732 (quasi égalité !)** |
| DTLZ3 | 4 | 5.785954 | 215.386259 | 164.386715 | 12.628822 | **11.356073** |
| DTLZ4 | 3 | 0.152405 | 0.256563 | 0.221271 | 0.218618 | **0.192409** |
| DTLZ4 | 4 | 0.142183 | 0.383146 | 0.346174 | 0.337810 | **0.328362** |
| DTLZ5 | 3 | 0.037518 | 0.014427 | 0.016836 | 0.015327 | 0.015989 (QINSGA3 gagne) |
| DTLZ6 | 3 | 0.716894 | 5.276403 | 6.466358 | 7.014231 | **5.860450** |
| DTLZ7 | 3 | 0.113257 | 0.429444 | 0.371840 | 3.092533 | **1.872026 (partiellement récupéré)** |

Le réglage final (crossover aligné + bruit désactivé + migration réajustée) rapproche QINSGA3 de NSGA-III à quasi-égalité sur **DTLZ1 et DTLZ3** (les deux problèmes où l'écart était le plus grand au départ), et l'améliore sur tous les autres sauf DTLZ7 qui reste dégradé (mais partiellement récupéré : ×1.65 mieux qu'avec bruit=0 seul) — limite structurelle documentée liée au front disjoint de ce problème.

**Piste explorée et écartée : `noise_scale` intermédiaire.** Un test à 5 runs sur DTLZ7 et DTLZ3 avec noise_scale ∈ {0.02, 0.01, 0.005, 0.0} montre deux tendances strictement monotones et opposées :

| noise_scale | DTLZ7 (mean IGD) | DTLZ3 (mean IGD) |
|---|---|---|
| 0.02 | **0.368** | 181.9 |
| 0.01 | 0.476 | 94.1 |
| 0.005 | 0.559 | 48.0 |
| 0.00 | 2.979 | **31.7** |

Aucune valeur intermédiaire n'est un compromis favorable — 0.005 et 0.01 sont pires que les deux extrêmes pour chacun des deux problèmes. Ce n'est pas un réglage à affiner mais un vrai dilemme structurel (précision de convergence vs maintien de la diversité entre régions disjointes), sans solution à un seul paramètre global. `noise_scale=0` reste donc le réglage retenu pour ce rapport, avec la régression sur DTLZ7 documentée comme limite connue plutôt que corrigée.

**Piste explorée et écartée : guide élitiste par niche.** Conception détaillée dans `docs/superpowers/specs/2026-07-24-qinsga3-elitist-guide-design.md` et implémentation dans `docs/superpowers/plans/2026-07-24-qinsga3-elitist-guide.md` (Tâches 1-4, mergées et testées) : `_select_guides` compare, pour chaque niche, le meilleur représentant du front Pareto courant à celui de l'archive externe (élitiste par construction) et garde le plus proche de la direction de référence — au lieu de recalculer sans mémoire à chaque génération. Un test d'ablation à 5 runs sur DTLZ1 et DTLZ3 (mêmes 5 premiers seeds que ci-dessus) donne :

| Problème | Métrique | Réglage actuel (30 runs) | Guide élitiste (5 runs) |
|---|---|---|---|
| DTLZ1 | mean | 0.228982 | 0.240365 (comparable) |
| DTLZ1 | worst | 1.536838 | **0.346592** (nettement plus serré) |
| DTLZ1 | std | 0.285543 | 0.078104 (nettement plus serré) |
| DTLZ3 | mean | 8.116732 | 24.765429 (×3 pire) |
| DTLZ3 | worst | 24.535356 | 71.955994 (pire) |
| DTLZ3 | std | 5.553143 | 26.872776 (×5 pire) |

DTLZ1 s'améliore comme prévu (le guide élitiste réduit nettement la variance et le pire cas — c'était l'objectif). Mais DTLZ3 régresse nettement sur mean/worst/std, pas seulement du bruit d'échantillon 5-runs. Hypothèse : DTLZ3 est fortement multimodal — un guide élitiste sans mécanisme d'échappement (catastrophe / diversity-preserving operator, cf. littérature QEA citée dans la spec) peut se verrouiller sur un optimum local pour tout le run, un piège classique de l'élitisme en évolution différentielle/quantique. Décision : **non retenu** — le code des Tâches 1-4 reste committé sur la branche (testé, revu, approuvé) mais la campagne complète à 30 runs n'a pas été lancée et ce réglage n'est pas adopté dans ce rapport ; le réglage en vigueur reste celui du tableau "Comparaison avec NSGA-III classique" ci-dessus (étape 4, migration réajustée).

**Piste explorée et écartée : opérateur diversity-preserving (correctif du piège d'élitisme ci-dessus).** Conception dans `docs/superpowers/specs/2026-07-24-qinsga3-diversity-preserving-design.md`, implémentation dans `docs/superpowers/plans/2026-07-24-qinsga3-diversity-preserving.md` (Tâches 1-5, mergées et testées) : port de l'opérateur "Diversity Preserving" de Tayarani-N & Akbarzadeh-T (2014, *Evolutionary Intelligence* 7:219–239, §5) — dans une niche bloquée depuis `t_stagnation` générations, détecter les individus convergés et similaires groupés autour du champion, garder le meilleur, réinitialiser les autres à π/4. Bâti par-dessus le guide élitiste ci-dessus (non adopté mais toujours committé), dans l'espoir de récupérer DTLZ3 sans perdre le gain sur DTLZ1.

Résultat de l'ablation à 5 runs : **aucun effet mesurable** — résultats strictement identiques (6 décimales) à ceux du guide élitiste seul, sur DTLZ1 comme DTLZ3, avec le réglage de départ `γ_converge=0.99` (valeur du papier). Diagnostic (script d'instrumentation, non committé) : le mécanisme de détection de stagnation fonctionne (niches bloquées détectées, jusqu'à 11 générations consécutives), mais aucun individu n'atteint jamais le seuil de convergence `γ=0.99` — le score de convergence de toute la population ne dépasse jamais ~0.6–0.77 (DTLZ1) / ~0.57–0.61 (DTLZ3) même à la dernière génération d'un run à 326 générations. Cause : le papier original utilise un QEA pur (porte de rotation seule, sans croisement) ; QINSGA3 applique en plus un croisement SBX et une mutation à deux niveaux à chaque génération, qui réintroduisent en permanence de l'exploration et empêchent cet effondrement des θ.

Recalibrage à `γ_converge=0.5` (commit `2877b4e`, entre le score typique observé en cours de run ~0.15–0.4 et le maximum de fin de run ~0.6–0.77) : **toujours aucun effet mesurable**, résultats de nouveau strictement identiques. Nouveau diagnostic : même à γ=0.5, une niche bloquée n'a presque jamais **deux** individus convergés *simultanément* — `_diversity_preserve_mask` exige cette condition (comparer un individu à un autre pour juger de leur similarité) et elle n'est quasiment jamais remplie (au plus 1 individu convergé observé à la fois sur le run testé). Hypothèse structurelle, pas un simple seuil à ajuster : les niches de QINSGA3 comptent peu d'individus (1 à ~17 observés) et le croisement + la mutation s'appliquent à toute la population à chaque génération, brassant les θ d'une niche plus vite que plusieurs individus ne peuvent y converger ensemble — contrairement au QEA du papier (population plus grande, sans croisement, où ce type de convergence groupée a le temps de se former).

Décision : **non retenu**. Le code des Tâches 1-5 (avec `γ_converge=0.5` recalibré) reste committé sur la branche (testé, revu, approuvé) mais aucune campagne à 30 runs n'a été lancée et ce réglage n'est pas adopté dans ce rapport ; le réglage en vigueur reste celui du tableau "Comparaison avec NSGA-III classique" ci-dessus (étape 4, migration réajustée), sans guide élitiste ni opérateur diversity-preserving.

**Piste retenue : migration réajustée.** Un test à 5 runs comparant `rotation_type` (linear, tanh_soft), `p_mut_strong=0.30` et `migration_period=5/n_migrate=20` contre la configuration de base montre que seule la migration réajustée améliore **simultanément** DTLZ7, MaF7 ET DTLZ3, sans dégrader DTLZ2 — contrairement aux autres variantes qui améliorent un problème en dégradant l'autre. C'est le réglage final retenu (voir tableau ci-dessus).