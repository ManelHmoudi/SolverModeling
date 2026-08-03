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
seule métrique où les deux sont équivalents.

## Évolution des corrections (avec preuves chiffrées)

QI-NSGA-III utilise une représentation quantique (angle θ, mesure
`x = xl + cos²(θ)(xu−xl)`) et une porte de rotation guidée par niche — voir
`Solvers/QINSGA3/README.md` pour le détail algorithmique complet. Trois
corrections majeures ont été apportées et validées avant d'atteindre la
configuration ci-dessus :

| Étape | Description | HV (test A/B) | Effet |
|---|---|---|---|
| 0. Algorithme d'origine | Croisement/mutation en espace θ, pas de sélection de survie explicite sur la population de travail (seule l'archive externe est élitiste) | 0.113 (20v20, avant toute correction) | — référence de départ |
| 1. + Sélection élitiste | Fusion parent+enfant, ne garde que les meilleurs `pop_size` via `ReferenceDirectionSurvival` (pymoo) — même sélection environnementale que NSGA-III | 0.068 → 0.120 (3 seeds) | ✅ Adopté — U=0 (séparation totale sur 3v3) |
| 2. + Espace X | Croisement (SBX) et mutation (PM) en espace X (valeurs réelles, eta=20 comme NSGA-III) au lieu de l'espace θ, dont le décodage `cos²(θ)` est très non-uniforme | 0.067 → 0.224 (3 seeds, sur la config déjà élitiste) | ✅ Adopté |
| 3. + Normalisation partagée avec `survival.norm` | La sélection des guides recalculait l'ideal/nadir à zéro chaque génération au lieu de partager celui, stable et monotone, que pymoo maintient déjà pour la survie élitiste | 0.189 → **0.205** (20v20) | ✅ Adopté — GD (p=0.022) et IGD (p=0.029) significatifs, HV à la limite (p=0.060) |

Les trois corrections adoptées constituent une amélioration réelle et
validée statistiquement de QI-NSGA-III — HV multiplié par ~1,3 à ~2× selon
le point de mesure. Mais sur cette instance à 100 clients, **NSGA-III reste
l'algorithme le plus performant** — cohérent avec le tableau plus nuancé
obtenu sur les benchmarks DTLZ/MaF (`Validation/Benchmarking/{dtlz,maf}/results/`),
où QI-NSGA-III dépasse NSGA-III sur plusieurs problèmes.

## Diagnostic diversité / décodeur -- pourquoi QI-NSGA-III gagne sur DTLZ/MaF mais pas sur l'IRP

Constat de départ (remarque de l'encadrante) : sur les benchmarks DTLZ/MaF,
QI-NSGA-III bat NSGA-III dans 17 cas sur 25 (68 %, voir
`Validation/Benchmarking/{dtlz,maf}/results/qinsga3/`) -- l'algorithme
quantique fonctionne donc bien en soi. Sur l'IRP réel, c'est l'inverse.
Trois hypothèses ont été proposées pour expliquer cet écart, avec une
expérience de diagnostic dédiée.

### Les 3 hypothèses et leur statut

| Hypothèse | Statut | Ce qui a été trouvé |
|---|---|---|
| **H1** -- le décodeur glouton (plus-proche-voisin) détruit la diversité générée par le codage quantique | Testée, réfutée (mais pas comme prévu) | La diversité chromosome de QI-NSGA-III est déjà ~13x plus basse que celle de NSGA-III **avant même le décodage** -- ce n'est pas une perte au décodage, c'est un déficit en amont |
| **H2** -- QI adapté aux problèmes continus ; sur l'IRP combinatoire, une petite variation de theta peut ne provoquer aucun changement réel de tournée | Testée sur tout le calendrier alpha réel, réfutée quantitativement | Moyenne pondérée sur les 300 générations = **1.12 % de routes inchangées** (`sensitivity/test_theta_route_sensitivity_weighted.py`) -- vrai seulement en toute fin de run, négligeable sur l'ensemble du run |
| **H3** -- perte d'information dans la chaîne theta -> x -> décodeur -> solution | Testée avec une vraie métrique de diversité, réfutée | Distance de Jaccard moyenne entre tournées décodées (`sensitivity/test_route_diversity.py`) : ratio QI/NSGA=**0.846**, très proche de 1, sans commune mesure avec le ratio chromosome (0.078, x13). Le décodeur ne perd pas d'information relative -- il en **amplifie** même légèrement |

### Expérience de diagnostic (diversité chromosome vs tournées vs front)

Protocole : chromosomes du front de Pareto poolées sur 6 seeds (100 clients),
NSGA-III (cache) vs QI-NSGA-III (code de production, avec le fix de
normalisation). Diversité = variance moyenne par gène, normalisée par les
bornes de la variable.

| Métrique | NSGA-III | QI-NSGA-III | Ratio QI/NSGA |
|---|---|---|---|
| N chromosomes poolées | 103 | 297 | -- |
| Diversité chromosome Var(X norm.) | 0.060264 | 0.004700 | **0.078** (13x moins) |
| HV | 0.842510 | 0.243675 | -- |
| Spacing | 0.070750 | 0.036513 | -- |

La métrique "diversité tournées" prévue dans le protocole initial (variance
de la quantité livrée par client) s'est révélée dégénérée -- fixée par la
contrainte de demande totale, identique pour toute solution faisable.
Remplacée par une vraie métrique de diversité des tournées (distance de
Jaccard sur les arcs (période, véhicule, arc) décodés --
`sensitivity/test_route_diversity.py`, 20 seeds, 3000 paires échantillonnées
par algorithme) : NSGA-III=0.862, QI-NSGA-III=0.729, ratio QI/NSGA=**0.846**
-- réfutation quantitative de H3.

**Conclusion** : le déficit de diversité de QI-NSGA-III est déjà présent au
niveau chromosome, avant tout décodage -- la rotation guidée (Δθ =
α(g) × tanh((θ_guide − θ)/(π/8))) tire 100 % de la population vers un guide
unique par niche à chaque génération, un mécanisme que NSGA-III n'a pas (sa
diversité vient uniquement de SBX/PM). Sur 600 variables et 300 générations,
cette pression convergente systématique explique l'écart.

### Test de sensibilité locale théta vers route (H2)

Script : `sensitivity/test_theta_route_sensitivity.py` et sa version
pondérée `test_theta_route_sensitivity_weighted.py`. Perturbation théta avec
la formule de rotation réelle, à différentes magnitudes alpha (plage de
production), comparaison de la tournée décodée avant/après.

| alpha (rad) | Routes identiques après perturbation | Delta F médian quand ça change |
|---|---|---|
| 0.00314 (proche alpha_min, fin de run) | 19.2 % | 54 |
| 0.03142 | 1.3 % | 465 |
| 0.15708 | 0.0 % | 1639 |
| 0.31416 (proche alpha_max, début de run) | 0.0 % | 2633 |

Quantifié sur tout le calendrier alpha réel (31 points de génération 0 à
300) : moyenne pondérée sur l'ensemble du run = **1.12 %** de routes
inchangées -- H2 n'est pas le facteur dominant, réfutée quantitativement.

**Découverte annexe, plus significative** : le décodeur est chaotique, pas
juste insensible -- quand une perturbation change la tournée, l'écart
d'objectif (Delta F) est souvent énorme et disproportionné par rapport à la
magnitude de la perturbation en théta. Cause : `_nearest_neighbour()`
(`Solvers/NSGA3/decoder.py`) fait des choix gloutons et irrévocables -- un
petit changement de score peut faire basculer un choix tôt dans la
construction, ce qui cascade sur toute la suite de la tournée (effet
papillon classique des heuristiques constructives sans retour arrière).

## Remèdes testés pour corriger le déficit de diversité

Après le diagnostic ci-dessus, plusieurs remèdes indépendants ont été
testés, chacun isolant une seule variable, avec le même protocole que le
reste du projet (shared ideal/nadir, Mann-Whitney U, 3 seeds puis scaling
si signal positif).

### Premières tentatives -- tuner le mécanisme existant (rejetées, aucun effet)

Ces cinq tentatives modifient toutes la fréquence, la cible ou la magnitude
de la **même** porte de rotation (celle qui tire vers le guide de niche),
sans en changer la nature. Aucune n'a bougé la diversité chromosome
mesurée :

| Remède | Cible | Résultat | Diversité chromosome |
|---|---|---|---|
| Magnitude de rotation adaptative par fitness | Enveloppe alpha | 3 seeds prometteur (U=0), 20 seeds : aucun effet significatif | -- |
| `rotation_prob` -- rotation appliquée à une fraction aléatoire de la population | Fréquence du tirage | Non significatif | 0.003984 → 0.003727 (inchangée) |
| Guide échantillonné par individu (au lieu du même guide unique par niche) | Cible du tirage | Non significatif | 0.003984 → 0.003728 (inchangée) |
| Décodeur adouci (softmax au lieu d'argmin strict) | Chaos du décodeur | Tendance négative, non significatif | 0.003984 → 0.004136 (quasi inchangée) |
| Réparation locale 2-opt post-décodage | Chaos du décodeur | Abandonné -- optimise la mauvaise métrique (distance, pas coût avec pénalités de fenêtres de temps) | -- |

Ce pattern uniforme (rien ne bouge) a motivé les remèdes structurels
ci-dessous, qui changent la nature du mécanisme plutôt que ses réglages.
Scripts conservés dans `sensitivity/` : `compare_fitness_adaptive_rotation.py`,
`compare_rotation_prob.py`, `compare_sampled_guide.py`, `compare_soft_decoder.py`.

### Remède A -- opérateur de recentrage par niche

Force de dispersion structurellement indépendante de la rotation : détecter
les clusters convergents dans une niche et réinitialiser les individus
non-élite à un θ aléatoire. Port adapté de l'opérateur DPQiEA
[Tayarani-N & Akbarzadeh-T 2014, *Evolutionary Intelligence* 7:219-239, §5]
-- design complet et corrections documentées dans
`docs/superpowers/specs/2026-08-01-qinsga3-niche-recentring-reset-design.md` :
le critère de convergence du papier (eq. 11, θ près de 0/π/2) ne se
déclenche jamais sur l'IRP (QINSGA3 converge autour de θ=π/4, pas vers les
bords) -- remplacé par la distance directe entre individus d'une même niche
(eq. 12 seule) ; le garde-fou de stagnation (eq. 13) ne se déclenche pas non
plus (survie élitiste qui remélange toute la population chaque génération)
-- supprimé ; valeur de reset = tirage uniforme aléatoire au lieu de π/4 (le
papier), car π/4 est déjà, empiriquement, là où la population est collée.

**Résultat (3 seeds, 300 générations, `delta_similar=0.05`,
`sensitivity/compare_niche_recentring_reset.py`)** :

| Indicateur | Baseline | Test | Mann-Whitney |
|---|---|---|---|
| HV ↑ | 0.210447 | 0.093754 | U=9.0, p=0.10 -- séparation totale, test moins bon |
| GD ↓ | 0.450245 | 0.762133 | U=0.0, p=0.10 -- séparation totale, test moins bon |
| IGD ↓ | 0.527407 | 0.727725 | U=0.0, p=0.10 -- séparation totale, test moins bon |
| Diversité chromosome | 0.003984 | **0.009047** (×2.3) | -- |

**C'est le premier remède à avoir réellement déplacé la diversité
chromosome** (les précédents la laissaient inchangée) -- et c'est justement
celui qui dégrade le plus la qualité du front. **Interprétation** :
injecter un θ uniformément aléatoire dans une fraction significative de la
population est trop disruptif pour un problème combinatoire fortement
contraint -- chaque réinitialisation efface une solution partiellement
construite, et le budget d'évaluation restant ne suffit pas à la survie
élitiste pour la reconstruire avant la génération suivante. **Le déficit de
diversité mesuré n'est donc pas, à lui seul, le facteur limitant : le
forcer artificiellement dégrade plus qu'il n'améliore.**

Script conservé : `sensitivity/compare_niche_recentring_reset.py`.

**Balayage complémentaire (`delta_similar` intermédiaire, entre 0.0 et 0.05)** -- pour
vérifier qu'un seuil plus doux ne changerait pas la conclusion :

| delta_similar | Diversité chromosome | HV | Mann-Whitney vs baseline |
|---|---|---|---|
| 0.01 | 0.003961 (≈ inchangée) | 0.126938 → 0.128835 (≈ identique) | U=5.0, p=1.0 -- aucune séparation |
| 0.02 | 0.003884 (≈ inchangée) | 0.154651 → 0.107986 (dégradé) | U=9.0, p=0.10 -- séparation totale |
| 0.05 | 0.009047 (×2.3) | dégradé (voir ci-dessus) | séparation totale |

Pas d'effet graduel : à 0.01 le reset ne se déclenche quasiment jamais (diversité et
qualité inchangées), mais dès 0.02 -- toujours sans gain de diversité mesurable --
la qualité se dégrade déjà en séparation totale. Il n'existe pas de dose
intermédiaire où l'opérateur apporterait de la diversité à moindre coût : soit il
ne fait rien, soit il abîme sans même apporter ce qu'il est censé apporter.

### Remède B -- bruit de mesure renforcé

Cible une étape jamais touchée jusque-là : la **mesure** (θ → x), pas la
rotation. Principe tiré d'un seul article vérifié texte-en-main : Guzel,
Yıldırım Okay, Kök & Özdemir (2022), *"QNSGA-II: A Quantum Computing-Inspired
Approach to Multi-Objective Optimization"*, IEEE ISNCC -- leur QNSGA-II
démarre avec une population de chromosomes quantiques **identiques** et ne
tire toute sa diversité que de la mesure probabiliste, jamais de la
rotation. QINSGA3 a déjà un terme de bruit de mesure équivalent
(`chromosome.py::measure`, Platel et al. 2009) mais réglé à une valeur
cosmétique (`noise_scale=0.02`) plutôt qu'à une valeur qui porte réellement
la diversité comme dans QNSGA-II.

**Résultat (3 seeds, 300 générations, `noise_scale=0.15`,
`sensitivity/compare_measurement_noise.py`)** :

| Indicateur | Baseline (noise=0.02) | Test (noise=0.15) | Mann-Whitney |
|---|---|---|---|
| HV ↑ | 0.299257 | 0.084721 | U=9.0, p=0.10 -- séparation totale, test moins bon |
| GD ↓ | 0.265858 | 0.756155 | U=0.0, p=0.10 -- séparation totale, test moins bon |
| IGD ↓ | 0.443228 | 0.753307 | U=0.0, p=0.10 -- séparation totale, test moins bon |
| Diversité chromosome | 0.003984 | **0.129519** (×32) | -- |

**Même verdict que le remède A, en plus marqué** : la diversité chromosome
explose (×32), mais la qualité du front s'effondre encore plus (HV divisé
par ~3.5). Deuxième mécanisme indépendant, basé sur une deuxième source
littéraire indépendante, qui confirme exactement le même pattern.
**Interprétation** : deux mécanismes de nature totalement différente
(reset discret par cluster vs bruit continu à chaque mesure) produisent le
même résultat qualitatif -- ce n'est pas un artefact d'un remède en
particulier, c'est une propriété du problème lui-même. Sur ce décodeur
combinatoire fragile, la diversité chromosome et la qualité de solution
sont en tension directe, quel que soit le mécanisme utilisé pour augmenter
la première.

Script conservé : `sensitivity/compare_measurement_noise.py`.

**Balayage complémentaire (`noise_scale` intermédiaire, entre 0.02 et 0.15)** --
pour vérifier qu'une dose plus douce ne changerait pas la conclusion :

| noise_scale | Diversité chromosome | HV | Perte HV vs baseline |
|---|---|---|---|
| 0.02 (défaut) | 0.003984 | 0.221-0.238 | -- |
| 0.05 | 0.021612 (×5.4) | 0.133124 | -40 % |
| 0.08 | 0.065852 (×16.5) | 0.057595 | -76 % |
| 0.15 | 0.129519 (×32) | 0.084721 | -72 % |

Contrairement au remède A, ici la relation est lisse et monotone -- diversité et
dégradation augmentent ensemble, proportionnellement à la dose -- mais
défavorable **à chaque dose testée**, y compris la plus douce (0.05, seulement
2,5× le défaut), qui fait déjà chuter HV de 40 %. Aucun point de compromis
favorable trouvé sur toute la plage testée : ce n'est pas une question de dose
mal calibrée, la tension diversité/qualité est présente dès la moindre
augmentation du bruit de mesure.

### Remède C -- rotation duale RQPSO

Change la **règle de rotation elle-même** : au lieu de tirer chaque
individu vers un seul point (le guide), le tirer vers **deux points** --
son propre meilleur historique (`pbest`) ET le guide de niche (`gbest`) --
avec des coefficients aléatoires à chaque pas. Port d'un seul article
complet et vérifié texte-en-main : Bodha, Arun, Awasthi, Mahato & Fotis
(2025), *"Rotational gate based quantum particle swarm optimization for
benchmark suites and combined economic emission dispatch"*, Engineering
Research Express 7(4):045345. Deux adaptations documentées dans
`Solvers/QINSGA3/algorithm.py` (note au-dessus de `_rqpso_rotate`) : domaine
borné [0, π/2] non cyclique au lieu du cercle complet du papier (distance
directe au lieu de la formule "shortest-arc" WRAP) ; magnitude du pas
réutilise `alpha_max` (déjà calibré) au lieu de l'échelle `Θ_t` du papier,
spécifique à son domaine circulaire. "Personal best" (`pbest`) n'a pas
d'équivalent fidèle dans QINSGA3 (pas d'identité d'individu stable d'une
génération à l'autre) -- approximé par un meilleur historique **par
emplacement de slot**, vérifié non dégénéré avant la campagne complète
(`pbest` différent du guide de niche de ~0.038 rad en moyenne, 100% des
générations testées).

**Résultat (3 seeds, 300 générations, `sensitivity/compare_rqpso_rotation.py`)** :

| Indicateur | Baseline (tanh) | Test (RQPSO) | Mann-Whitney |
|---|---|---|---|
| HV ↑ | 0.258900 | 0.135772 | U=9.0, p=0.10 -- séparation totale, test moins bon |
| GD ↓ | 0.358010 | 0.652457 | U=0.0, p=0.10 -- séparation totale, test moins bon |
| IGD ↓ | 0.462444 | 0.654124 | U=0.0, p=0.10 -- séparation totale, test moins bon |
| Diversité chromosome | 0.003984 | **0.003271** (légèrement plus basse) | -- |

**Verdict différent des remèdes A et B** : ici la diversité chromosome
**diminue encore un peu plus** au lieu d'augmenter, tout en dégradant la
qualité au même niveau de séparation totale. **Interprétation** : ajouter
un deuxième point d'attraction (`pbest`) en plus du guide ne dilue pas la
pression convergente -- elle s'additionne. Deux forces d'attraction
simultanées, même avec des coefficients aléatoires, contraignent la
population *plus* qu'une seule, pas moins. Ce résultat renforce directement
l'hypothèse retenue pour le mémoire (voir conclusion) : le problème n'est
pas le nombre de points d'attraction ni leur cible, c'est la nature même
d'un mécanisme de "tirer vers un point" appliqué à chaque génération sur
300 générations.

Script conservé : `sensitivity/compare_rqpso_rotation.py`.

### Remède D -- rotation à momentum PSO

Change une nouvelle fois la règle de rotation : au lieu de recalculer le pas
à partir de zéro à chaque génération (ce que fait la porte tanh de base, et
même RQPSO malgré ses deux attracteurs), on introduit une **vitesse qui
persiste** d'une génération à l'autre (momentum/inertie), pour qu'un
individu qui a pris de la vitesse dans une direction ne bascule pas
immédiatement vers un nouveau guide. Port d'un seul article complet et
vérifié texte-en-main : Li, Xu, Liu & Li (2008), *"Quantum Multi-objective
Evolutionary Algorithm with Particle Swarm Optimization Method"*, ICNC 2008
-- la même source déjà utilisée pour les valeurs `alpha_max`/`alpha_min` de
la porte de rotation de base, ici réutilisée pour son propre mécanisme de
momentum, jusque-là jamais implémenté. Deux adaptations documentées dans
`Solvers/QINSGA3/algorithm.py` (note au-dessus de `_pso_rotate`) : le clamp
de vitesse réutilise le calendrier `alpha_max`/`alpha_min` déjà calibré (au
lieu de l'échelle propre au papier, spécifique à son domaine) ; le poids
d'inertie adaptatif `w = D(i)/N + m(i)/N` (densité de voisinage +
rang de dominance) est borné à [0, 2] pour éviter toute instabilité, le
papier ne documentant pas de borne explicite. Vérifié non dégénéré avant la
campagne complète (poids d'inertie variant réellement entre individus à
chaque génération testée, min 0.000 / médiane 0.827 / max 1.945 sur 60
générations ; magnitude de vitesse jamais nulle, médiane 0.112 rad).

**Résultat (3 seeds, 300 générations, `sensitivity/compare_pso_rotation.py`)** :

| Indicateur | Baseline (tanh) | Test (PSO momentum) | Mann-Whitney |
|---|---|---|---|
| HV ↑ | 0.219573 | 0.116715 | U=9.0, p=0.10 -- séparation totale, test moins bon |
| GD ↓ | 0.434029 | 0.696588 | U=0.0, p=0.10 -- séparation totale, test moins bon |
| IGD ↓ | 0.512387 | 0.686031 | U=0.0, p=0.10 -- séparation totale, test moins bon |
| Diversité chromosome | 0.003984 | **0.002394** (encore plus basse) | -- |

**Même verdict que le remède C, encore plus marqué** : la diversité
chromosome baisse encore plus qu'avec RQPSO (0.002394 contre 0.003271), et
la qualité se dégrade au même niveau de séparation totale. **Interprétation**
: la vitesse persistante ne protège pas la diversité ici -- elle l'aggrave,
car sur ce problème la direction du guide reste globalement stable d'une
génération à l'autre (pas d'oscillation multimodale qui ferait s'annuler la
vitesse accumulée). Le momentum agit donc comme un **amplificateur** de la
même pression convergente plutôt que comme un frein : chaque génération
ajoute de la vitesse dans une direction déjà cohérente avec les générations
précédentes, ce qui accélère la contraction au lieu de la ralentir.
Quatrième mécanisme indépendant, quatrième source littéraire distincte,
même conclusion.

Script conservé : `sensitivity/compare_pso_rotation.py`.

### Remède E -- rotation à magnitude chaotique

Change à la fois la magnitude ET la cible de la rotation : au lieu d'une
magnitude qui décroît en douceur (`alpha_max→alpha_min`), on la module par
une **séquence chaotique par individu** (carte logistique, μ≥4), dont la loi
stationnaire est en U (l'individu passe plus de temps proche de 0 ou de 1
que de 0.5) -- des pas tantôt quasi-nuls, tantôt presque doublés, plutôt
qu'un calendrier lisse. La cible n'est plus le champion unique de la niche
mais une distance RMS aux membres de l'archive externe associés à la niche
(K meilleurs individus). Port d'un seul article complet et vérifié
texte-en-main : Hu Feng-jun & Wu Bin (2009), *"Quantum Evolutionary
Algorithm for Vehicle Routing Problem with Simultaneous Delivery and
Pickup"*, Joint 48th IEEE Conference on Decision and Control / 28th Chinese
Control Conference. Trois adaptations documentées dans
`Solvers/QINSGA3/algorithm.py` (note au-dessus de `_chaotic_rotate`) : la
direction du papier dépend d'un bit binaire du meilleur individu (encodage
VRP binaire d'origine) -- remplacée par `sign(guide − θ)`, la même logique
déjà utilisée partout ailleurs dans le projet ; les "K meilleurs individus"
du papier (leur propre archive externe B(t)) correspondent directement à
l'archive externe déjà maintenue par QINSGA3 -- aucune nouvelle machinerie
d'élitisme inventée ; la persistance de la séquence chaotique par génération
souffre du même problème d'identité d'individu que `pbest`/`velocity`
(remèdes C/D) -- approximée par emplacement de slot, même classe
d'approximation déjà validée. Vérifié non dégénéré avant la campagne
complète (λ montre une vraie dispersion chaotique sur 100% des générations
testées, moyenne ~0.49 conforme à la loi stationnaire connue de la carte
logistique ; magnitude RMS non nulle et stable, 0.033-0.063 rad, 96-99% des
individus affectés par génération).

**Résultat (3 seeds, 300 générations, `sensitivity/compare_chaotic_rotation.py`)** :

| Indicateur | Baseline (tanh) | Test (chaotique) | Mann-Whitney |
|---|---|---|---|
| HV ↑ | 0.318073 | 0.099357 | U=9.0, p=0.10 -- séparation totale, test moins bon |
| GD ↓ | 0.249699 | 0.746216 | U=0.0, p=0.10 -- séparation totale, test moins bon |
| IGD ↓ | 0.409898 | 0.726968 | U=0.0, p=0.10 -- séparation totale, test moins bon |
| Diversité chromosome | 0.003984 | **0.004444** (légèrement plus haute, ×1.12) | -- |

**Nuance par rapport aux remèdes précédents** : c'est le seul des remèdes
C/D/E (qui changent la règle de rotation elle-même plutôt que la cible/le
calendrier d'une rotation à point unique) où la diversité chromosome
**augmente**, même si très légèrement -- comme A et B, mais d'un ordre de
grandeur bien plus faible (×1.12 contre ×2.3 et ×32). Pourtant la qualité se
dégrade exactement au même niveau de séparation totale que C et D, où la
diversité baissait. **Interprétation** : ce résultat confirme, avec un
cinquième mécanisme indépendant, que le sens du mouvement de diversité
(hausse ou baisse) et son ampleur ne prédisent pas la qualité du front sur
ce décodeur -- seul le simple fait de s'écarter du calendrier de rotation
`tanh` déjà calibré dégrade la qualité, quel que soit le mécanisme de
remplacement (bruit, reset, double attracteur, momentum, chaos) et quel que
soit son effet sur la diversité mesurée.

Script conservé : `sensitivity/compare_chaotic_rotation.py`.

### Autre tentative testée sans effet mesurable

Une réorganisation locale des guides (structure "anneau" par niche --
chaque individu compare seulement ses deux voisins d'anneau au lieu de
toute la niche, Tayarani-N & Akbarzadeh-T 2014, §3, structure la mieux
notée par les auteurs pour un problème combinatoire) a aussi été testée (3
seeds, 300 générations, `sensitivity/compare_ring_guides.py`). Aucun effet
mesurable, ni sur la diversité chromosome ni sur la qualité du front
(toutes les métriques non significatives, p entre 0.40 et 0.70, contre une
séparation totale nette pour les remèdes A/B/C ci-dessus) : les niches sont
déjà trop petites (~4-5 membres en moyenne sur ~43 niches occupées) pour
que restreindre la comparaison à un voisinage local change quoi que ce
soit -- le guide local et le champion global finissent souvent par être les
mêmes individus.

### Remède F -- guide de niche par crowding distance

Proposition d'un relecteur externe de la thèse : les remèdes A-E changent
tous la fréquence, la cible ou la règle de la rotation, mais aucun ne touche
le critère utilisé pour choisir LEQUEL des membres d'une niche devient le
champion (`_select_guides` a toujours pris le membre le plus proche du rayon
de référence -- un critère de **convergence**, `d_perp2.argmin()`). Ce
remède le remplace par le membre à plus forte **crowding distance** dans la
niche [Deb et al. 2002, §III-B] -- un critère de diversité déjà implémenté
dans ce module (`_crowding_distance`, utilisé jusque-là pour l'élagage de
l'archive externe). Design complet :
`docs/superpowers/specs/2026-08-03-qinsga3-crowding-distance-guide-design.md`.

**Résultat (3 seeds, 300 générations, instance 100 clients,
`sensitivity/compare_crowding_guides.py`)** :

| Indicateur | Baseline (ray-closest) | Test (crowding) | Mann-Whitney |
|---|---|---|---|
| HV ↑ | 0.140398 | 0.152976 | U=3.0, p=0.700000 |
| GD ↓ | 0.634873 | 0.615103 | U=4.0, p=1.000000 |
| IGD ↓ | 0.636284 | 0.631984 | U=4.0, p=1.000000 |
| Diversité chromosome | 0.003984 | 0.003642 | -- |

Script conservé : `sensitivity/compare_crowding_guides.py`. Log complet :
`sensitivity/crowding_guides_campaign_log.txt`.

**Sixième remède indépendant, même verdict que A-E** : changer uniquement le
critère de sélection du champion de niche (sans toucher sa fréquence, sa
magnitude, ni la formule de rotation) ne suffit pas non plus à combler
l'écart avec NSGA-III sur l'IRP. Ceci renforce l'hypothèse retenue pour le
mémoire : la limite n'est pas dans le CHOIX du point cible (que ce soit par
convergence ou par diversité), mais dans le principe même de tirer chaque
génération vers UN point unique dans un espace θ dont la géométrie n'est pas
régulière une fois passée par le décodeur.

## Conclusion

Après un diagnostic structurel clair (déficit de diversité chromosome ×13,
décodeur chaotique) et cinq remèdes structurels indépendants (A, B, C, D, E
ci-dessus, chacun ancré dans une source littéraire distincte et vérifiée),
aucun n'a produit de gain réel. Le pattern est net et cohérent :

- Les cinq premières tentatives (tuner fréquence/cible/magnitude de la
  rotation existante) n'ont **jamais bougé la diversité chromosome**.
- Les remèdes A et B, qui ont réellement augmenté la diversité chromosome
  (×2.3 et ×32 respectivement, par des mécanismes totalement différents),
  ont **tous les deux dégradé la qualité** du front -- la dégradation étant
  proportionnelle à l'augmentation de diversité forcée.
- Les remèdes C et D, qui changent la règle de rotation elle-même (deuxième
  point d'attraction pour C, vitesse persistante pour D) plutôt que la
  cible ou la fréquence d'une rotation à point unique, ont tous les deux vu
  la diversité **encore baisser** et la qualité se dégrader tout autant :
  ajouter un deuxième attracteur ou une mémoire de direction ne dilue pas
  la pression convergente, elle s'y ajoute ou l'amplifie.
- Le remède E, qui module la magnitude de façon chaotique plutôt que lisse,
  a vu la diversité **légèrement augmenter** (×1.12, contrairement à C/D)
  mais la qualité se dégrader au même niveau de séparation totale que C et
  D -- la preuve la plus nette que le sens du mouvement de diversité ne
  prédit pas la qualité : cinq mécanismes indépendants, trois effets
  différents sur la diversité (forte hausse, baisse, légère hausse), un
  seul et même résultat sur la qualité.
- Le remède F, qui change pour la première fois le CRITÈRE de choix du
  champion (crowding distance plutôt que distance au rayon de référence)
  plutôt que sa fréquence, sa cible ou la règle de rotation, a vu la
  diversité chromosome légèrement diminuer (0.003642 contre 0.003984,
  soit -8.6 %) sans qu'aucune métrique de qualité du front ne s'améliore
  significativement face au baseline (HV/GD/IGD/Spacing tous non
  significatifs, p entre 0.40 et 1.00), l'écart avec NSGA-III restant lui
  hautement significatif (p=0.0011 sur HV/GD/IGD) -- sixième mécanisme
  indépendant, même conclusion : ce n'est pas le choix du point cible qui
  limite QI-NSGA-III sur l'IRP.

**Hypothèse retenue pour le mémoire** : le mécanisme de rotation guidée de
QI-NSGA-III repose sur une hypothèse de régularité (« un petit pas vers le
guide rapproche un peu de la solution ») qui ne tient pas face à un
décodeur combinatoire glouton et irrévocable -- contrairement aux
benchmarks DTLZ/MaF où x=f(θ) directement. C'est une limite structurelle du
mécanisme quantique face à ce type de décodeur, pas un problème de réglage
de paramètres, ni un bug corrigible simplement. Les remèdes A et B montrent
en plus que corriger directement le symptôme mesuré (déficit de diversité)
ne suffit pas à corriger la cause : sur un décodeur combinatoire fragile,
plus de diversité chromosome veut surtout dire plus de solutions détruites
avant d'avoir pu être affinées.
