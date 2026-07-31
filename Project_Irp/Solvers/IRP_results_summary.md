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

## Diagnostic diversite / decodeur -- pourquoi QI-NSGA-III gagne sur DTLZ/MaF mais pas sur l'IRP

Constat de depart (remarque de l'encadrante) : sur les benchmarks DTLZ/MaF,
QI-NSGA-III bat NSGA-III dans 17 cas sur 25 (68 %, voir
`Validation/Benchmarking/{dtlz,maf}/results/qinsga3/`) -- l'algorithme
quantique fonctionne donc bien en soi. Sur l'IRP reel, c'est l'inverse.
Trois hypotheses ont ete proposees pour expliquer cet ecart, avec une
experience de diagnostic dediee.

### Les 3 hypotheses et leur statut

| Hypothese | Statut | Ce qui a ete trouve |
|---|---|---|
| **H1** -- le decodeur glouton (plus-proche-voisin) detruit la diversite generee par le codage quantique | Testee, refutee (mais pas comme prevu) | La diversite chromosome de QI-NSGA-III est deja ~13x plus basse que celle de NSGA-III **avant meme le decodage** (voir ci-dessous) -- ce n'est pas une perte au decodage, c'est un deficit en amont |
| **H2** -- QI adapte aux problemes continus ; sur l'IRP combinatoire, une petite variation de theta peut ne provoquer aucun changement reel de tournee | Testee directement, confirmee partiellement | Voir "Test de sensibilite locale" ci-dessous : vrai seulement en toute fin de run (alpha faible), negligeable sur l'essentiel du run |
| **H3** -- perte d'information dans la chaine theta -> x -> decodeur -> solution | Testee indirectement, pas de preuve solide | Ratio de structures de routes uniques = 100 % pour les deux algorithmes (pas de collision totale), mais test faible |

### Experience de diagnostic (diversite chromosome vs tournees vs front)

Protocole : chromosomes du front de Pareto poolees sur 6 seeds (100 clients),
NSGA-III (cache) vs QI-NSGA-III (code de production, avec le fix de
normalisation). Diversite = variance moyenne par gene, normalisee par les
bornes de la variable.

| Metrique | NSGA-III | QI-NSGA-III | Ratio QI/NSGA |
|---|---|---|---|
| N chromosomes poolees | 103 | 297 | -- |
| Diversite chromosome Var(X norm.) | 0.060264 | 0.004700 | **0.078** (13x moins) |
| HV | 0.842510 | 0.243675 | -- |
| Spacing | 0.070750 | 0.036513 | -- |

La metrique "diversite tournees" (variance de la quantite livree par
client, sommee sur les periodes) prevue dans le protocole initial s'est
revelee degeneree : cette somme est fixee par la contrainte de satisfaction
de la demande totale, donc identique pour toute solution faisable quel que
soit l'algorithme -- elle ne mesure rien d'utile et a ete abandonnee au
profit du ratio de structures de routes uniques (100 % pour les deux
algorithmes, voir tableau H1-H3 ci-dessus).

**Conclusion** : le deficit de diversite de QI-NSGA-III est deja present au
niveau chromosome, avant tout decodage -- la rotation guidee (Delta theta =
alpha(g) x tanh((theta_guide - theta)/(pi/8))) tire 100 % de la population
vers un guide unique par niche a chaque generation, un mecanisme que
NSGA-III n'a pas (sa diversite vient uniquement de SBX/PM). Sur 600
variables et 300 generations, cette pression convergente systematique
explique l'ecart.

### Test de sensibilite locale theta vers route (H2)

Script : `sensitivity/test_theta_route_sensitivity.py`. Perturbation
theta vers theta perturbe avec la formule de rotation reelle (tanh, guide =
une autre solution reelle du pool), a differentes magnitudes alpha (plage
de production, 0.001pi a 0.10pi), puis comparaison de la tournee decodee
avant/apres.

| alpha (rad) | Routes identiques apres perturbation | Delta F median quand ca change |
|---|---|---|
| 0.00314 (proche alpha_min, fin de run) | 19.2 % (15/78) | 54 |
| 0.03142 | 1.3 % | 465 |
| 0.15708 | 0.0 % | 1639 |
| 0.31416 (proche alpha_max, debut de run) | 0.0 % | 2633 |

H2 confirmee seulement dans une fenetre etroite (alpha tres faible, fin de
run) -- pas le facteur dominant sur l'ensemble du run. Decouverte annexe,
plus significative : le decodeur est chaotique, pas juste insensible --
quand une perturbation change la tournee, l'ecart d'objectif (Delta F)
est souvent enorme et disproportionne (jusqu'a mediane 2633, pour un f1
typique de 15000-18000) par rapport a la magnitude de la perturbation en
theta. Cause : `_nearest_neighbour()` (`Solvers/NSGA3/decoder.py`) fait des
choix gloutons et irrevocables -- un petit changement de score peut
faire basculer un choix tot dans la construction, ce qui cascade sur toute
la suite de la tournee (effet papillon classique des heuristiques
constructives sans retour arriere).

### Quatre remedes testes et rejetes (fin juillet - debut aout 2026)

Chaque remede isole une seule variable, valide avec le meme protocole que
le reste du projet (shared ideal/nadir, Mann-Whitney U, 3 seeds puis
scaling si signal positif).

| Remede | Cible | Resultat (3 seeds) | Diversite chromosome | Decision |
|---|---|---|---|---|
| Magnitude de rotation adaptative par fitness (Kumar et al. 2026) | Enveloppe alpha | 3 seeds prometteur (U=0) puis 20 seeds : aucun effet significatif | -- | Rejete (voir tableau precedent) |
| rotation_prob -- rotation appliquee a une fraction aleatoire de la population par generation au lieu de 100 % | Frequence du tirage vers le guide | HV/GD/IGD legerement mieux, p non significatif | 0.003984 vers 0.003727 (inchangee) | Rejete -- reduire la frequence ne change pas la convergence cumulee sur 300 generations |
| Guide echantillonne par individu (au lieu du meme guide unique par niche) | Cible du tirage | HV/GD/IGD legerement mieux, p non significatif | 0.003984 vers 0.003728 (inchangee) | Rejete -- meme verdict que rotation_prob, confirme que le guidage n'est pas le facteur dominant |
| Decodeur adouci (selection softmax au lieu d'argmin strict dans _nearest_neighbour) | Chaos du decodeur | HV/GD pires (p non significatif mais tendance negative) | 0.003984 vers 0.004136 (quasi inchangee) | Rejete -- reduit le chaos mesure de 24 % (test de sensibilite) mais ne se traduit par aucun gain, voire une legere degradation |
| Reparation locale 2-opt post-decodage | Chaos du decodeur (construction) | Abandonne avant test complet | -- | Rejete -- le 2-opt minimise la distance seule ; sur 10 solutions verifiees, f3 (temps de trajet) s'ameliore toujours mais f1 (cout, incluant les penalites de fenetres de temps) se degrade systematiquement (+90 a +735) -- optimise le mauvais critere pour ce probleme |

Scripts conserves dans `sensitivity/` : `compare_fitness_adaptive_rotation.py`,
`compare_rotation_prob.py`, `compare_sampled_guide.py`,
`compare_soft_decoder.py`, `test_theta_route_sensitivity.py`.
`compare_2opt_repair.py` n'a jamais atteint un etat valide (bug de fond
identifie avant le test complet) -- supprime.

### Conclusion de ce chapitre

Apres un diagnostic structurel clair (deficit de diversite chromosome x13,
decodeur chaotique) et quatre tentatives de remede independantes, aucune
n'a produit de gain solide. Le pattern est coherent : le guidage
(frequence, cible) n'est pas le levier determinant, et lisser le decodeur
aide un peu sur le papier (-24 % de chaos mesure) sans se traduire en gain
reel. Hypothese retenue pour le memoire : le mecanisme de rotation guidee
de QI-NSGA-III repose sur une hypothese de regularite (« un petit pas vers
le guide rapproche un peu de la solution ») qui ne tient pas face a un
decodeur combinatoire glouton et irrevocable -- contrairement aux
benchmarks DTLZ/MaF ou x=f(theta) directement. C'est une limite
structurelle du mecanisme quantique face a ce type de decodeur, pas un
probleme de reglage de parametres (largement ecarte egalement, voir
tableau "Pistes testees et rejetees" plus haut) ni un bug corrigible
simplement.
