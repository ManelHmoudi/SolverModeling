# Résultats NSGA-III vs QI-NSGA-III sur l'IRP réel

Contrairement aux benchmarks synthétiques (DTLZ/MaF, voir
`Validation/Benchmarking/{dtlz,maf}/results/`), cette comparaison porte sur
le **vrai problème IRP** (`data/instance_100_clients.json`, 100 clients,
101 nœuds, 5 périodes, 6 véhicules) — l'instance la plus grande et la plus
exigeante disponible, choisie pour distinguer les deux algorithmes sur un
espace de décision réaliste (600 variables).

## Protocole

- 20 runs indépendants par algorithme, mêmes 20 seeds pour les deux
  (`[42, 137, 271, 491, 613, 733, 857, 977, 1009, 1123, 1249, 1373, 1499,
  1609, 1733, 1871, 1997, 2113, 2237, 2351]`).
- `pop_size=200`, `n_gen=300` — budget de recherche identique pour les deux.
- Indicateurs (HV, GD, IGD, Spacing) calculés avec un **ideal/nadir global
  partagé entre les deux algorithmes** (pas le ideal/nadir propre à chaque
  run, qui n'est pas comparable d'un algorithme à l'autre — voir
  `sensitivity/compare_qinsga3_vs_nsga3.py`), et un test de Mann-Whitney U
  pour la significativité statistique.
- Chromosomes mis en cache (`Solvers/NSGA3/nsga3_chromosomes.json`,
  `Solvers/QINSGA3/qinsga3_chromosomes.json`), re-décodés à chaque
  comparaison contre les données d'instance courantes.

## Résultat final validé

| Indicateur | NSGA-III | QI-NSGA-III | Mann-Whitney | Verdict |
|---|---|---|---|---|
| HV ↑ | **0.526600** (σ=0.172) | 0.205470 (σ=0.028) | p<0.001 | NSGA-III significativement meilleur |
| GD ↓ | **0.169105** (σ=0.085) | 0.457975 (σ=0.059) | p<0.001 | NSGA-III significativement meilleur |
| IGD ↓ | **0.361349** (σ=0.035) | 0.535515 (σ=0.035) | p<0.001 | NSGA-III significativement meilleur |
| Spacing ↓ | 0.047252 | 0.036220 | p=0.126 | Pas de différence significative |

**Sur cette instance, NSGA-III reste significativement meilleur que
QI-NSGA-III sur les 3 indicateurs de qualité**, malgré les corrections
apportées (voir ci-dessous) — Spacing (répartition des solutions) est la
seule métrique où les deux sont équivalents. Tableau mis à jour après
l'étape 3 (normalisation partagée, voir ci-dessous) ; les chiffres évoluent
légèrement d'une campagne à l'autre à cause du ideal/nadir partagé recalculé
à chaque comparaison (voir note plus bas), sans changer le verdict.

## Évolution des corrections (avec preuves chiffrées)

QI-NSGA-III utilise une représentation quantique (angle θ, mesure
`x = xl + cos²(θ)(xu−xl)`) et une porte de rotation guidée par niche — voir
`Solvers/QINSGA3/README.md` pour le détail algorithmique complet. Deux
corrections majeures ont été apportées et validées avant d'atteindre la
configuration ci-dessus :

| Étape | Description | HV (test A/B) | Effet |
|---|---|---|---|
| 0. Algorithme d'origine | Croisement/mutation en espace θ, pas de sélection de survie explicite sur la population de travail (seule l'archive externe est élitiste) | 0.113 (20v20, avant toute correction) | — référence de départ |
| 1. + Sélection élitiste | Fusion parent+enfant, ne garde que les meilleurs `pop_size` via `ReferenceDirectionSurvival` (pymoo) — même sélection environnementale que NSGA-III | 0.068 → 0.120 (3 seeds) | ✅ Adopté — U=0 (séparation totale sur 3v3) |
| 2. + Espace X | Croisement (SBX) et mutation (PM) en espace X (valeurs réelles, eta=20 comme NSGA-III) au lieu de l'espace θ, dont le décodage `cos²(θ)` est très non-uniforme | 0.067 → 0.224 (3 seeds, sur la config déjà élitiste) | ✅ Adopté |
| → Validation finale (20v20) | Config confirmée (élitiste + espace X + α_max=0.10π + rotation="tanh") | **0.182** puis **0.144** (voir note ci-dessous) | Résultat étape 2 |
| 3. + Normalisation partagée avec `survival.norm` | La sélection des guides recalculait l'ideal/nadir à zéro chaque génération au lieu de partager celui, stable et monotone, que pymoo maintient déjà pour la survie élitiste — deux normalisations incohérentes du même concept dans la même génération | 0.189 → **0.205** (20v20, ancien comportement vs corrigé) | ✅ Adopté — GD (p=0.022) et IGD (p=0.029) significatifs, HV à la limite (p=0.060), Spacing inchangé (p=0.365, cohérent : correctif de convergence, pas de diversité) |

**Note sur la variation 0.182 → 0.144** : ce nombre dépend du ideal/nadir
partagé, recalculé à chaque comparaison à partir des chromosomes alors en
cache pour les deux algorithmes — une régénération de cache (même config,
mêmes seeds) peut légèrement déplacer ce nadir partagé et donc la valeur de
HV normalisée, sans changer le verdict (NSGA-III reste significativement
meilleur dans les deux cas, p<0.001). Le tableau "Résultat final validé"
ci-dessus reflète l'état des caches actuellement sur disque.

## Pistes testées et rejetées

Chacune de ces pistes a été testée avec la méthodologie ci-dessus
(ideal/nadir partagé quand comparé à NSGA-III) ou par comparaison interne
entre variantes QI-NSGA-III (précisé à chaque fois) :

| Piste | Résultat | Décision |
|---|---|---|
| `eta_cross` en espace θ (5 vs 20) | Aucune différence significative entre les deux (les deux nettement sous NSGA-III) | ❌ Non pertinent — le problème n'était pas la largeur du croisement mais l'espace où il opère (voir correction 2 ci-dessus) |
| Garder l'archive externe vs la retirer (avec sélection élitiste) | Avec archive : HV=0.137 ; sans archive : HV=0.087 (3 seeds) | ✅ Archive gardée |
| Initialisation dispersée (θ ~ U(0,π/2), comme NSGA-III) vs concentrée (θ=π/4, actuelle) | Concentrée : HV=0.142 ; dispersée : HV=0.065 (3 seeds) — hypothèse inversée : la concentration aide (probablement plus proche de la faisabilité sur un problème très contraint) | ❌ Initialisation actuelle gardée telle quelle |
| Rotation adaptative par niche (règle du 1/5e de Rechenberg, pas de rotation modulé selon amélioration réelle) | HV base=0.161 vs adaptatif=0.168 — non significatif (p=1.0 sur HV/GD/IGD) | ❌ Non adopté, effet neutre |
| `alpha_max=0.01π` (au lieu de 0.10π) | +3.8 à +9.3% de HV **en comparaison interne uniquement** (QI-NSGA-III vs lui-même, sans NSGA-III) | ⚠️ Piste trompeuse, voir ligne suivante |
| `rotation_type="tanh_soft"` (au lieu de "tanh") | +2.9% de HV **en comparaison interne uniquement** | ⚠️ Piste trompeuse, voir ligne suivante |
| Combinaison des deux ci-dessus, **validée contre NSGA-III** (20v20) | HV=0.169 vs NSGA-III=0.474 sur ce run — pas d'amélioration réelle par rapport à la config de production (aurait même légèrement dégradé GD/IGD) | ❌ Non adopté — piège classique de comparaison interne (chaque variante mesurée seulement contre elle-même) qui ne se confirme pas une fois normalisée sur l'échelle partagée avec NSGA-III |
| Magnitude de rotation adaptative par fitness (`Δθ = η·tanh((f_best−f)/(\|f\|+ε))`, Kumar, Solanki, Jhariya, Shrivastava & Gupta 2026, *Scientific Reports*, EAH-QNSGA-II — adapté ici via la distance perpendiculaire au rayon de référence comme proxy de fitness) | 3 seeds : séparation totale sur HV (U=0, p=0.10 — plancher statistique à cet échantillon) laissait espérer un effet réel. 20 seeds : **aucune différence significative** sur les 4 métriques (HV p=0.925, GD p=0.172, IGD p=0.409, Spacing p=0.091) — le signal à 3 seeds était du bruit | ❌ Non adopté — script gardé (`sensitivity/compare_fitness_adaptive_rotation.py`) comme trace, aucune modification de production |

## Conclusion

Les trois corrections adoptées (sélection élitiste, variation en espace X,
normalisation partagée) constituent une amélioration réelle et validée
statistiquement de QI-NSGA-III — HV multiplié par ~1,3 à ~2× selon le point
de mesure entre les deux premières corrections, puis un gain supplémentaire
significatif sur GD/IGD (et HV à la limite) avec la troisième. Mais sur
cette instance à 100 clients, **NSGA-III reste l'algorithme le plus
performant** pour résoudre l'IRP lui-même — un résultat à assumer tel quel
plutôt qu'à forcer, cohérent avec le tableau plus nuancé obtenu sur les
benchmarks DTLZ/MaF (voir `Validation/Benchmarking/{dtlz,maf}/results/`),
où QI-NSGA-III dépasse NSGA-III sur plusieurs problèmes — et où la
normalisation partagée améliore aussi les résultats sur la plupart des
problèmes DTLZ/MaF (DTLZ7 et MaF7 en M3 basculent même en faveur de
QI-NSGA-III, qui perdait auparavant sur ces deux problèmes).
