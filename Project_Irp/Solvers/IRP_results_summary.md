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

**Sixième remède indépendant, même verdict que A-E** : changer uniquement le
critère de sélection du champion de niche (sans toucher sa fréquence, sa
magnitude, ni la formule de rotation) ne suffit pas non plus à combler
l'écart avec NSGA-III sur l'IRP. Ceci renforce l'hypothèse retenue pour le
mémoire : la limite n'est pas dans le CHOIX du point cible (que ce soit par
convergence ou par diversité), mais dans le principe même de tirer chaque
génération vers UN point unique dans un espace θ dont la géométrie n'est pas
régulière une fois passée par le décodeur. Contrairement aux remèdes A-E
(tous en séparation totale, U=0 ou U=9 sur 3v3 -- le test clairement pire),
le remède F ne dégrade rien : distributions chevauchantes (U=3.0-7.0), les
quatre métriques penchant nominalement en faveur du test sans signification
statistique -- un résultat neutre, pas nuisible.

**Réserve méthodologique** : `_crowding_distance` assigne une valeur
infinie aux membres extrêmes de chaque objectif -- avec M=4 objectifs et des
niches de 1 à 5 membres (taille typique sur l'IRP, voir le remède "anneau"
ci-dessus), le critère dégénère alors en un simple tie-break positionnel
(premier indice) plutôt qu'un vrai signal de diversité. Les niches à 1
membre ne sont pas concernées. Ce remède a donc surtout comparé "convergence
vs choix positionnel arbitraire" plutôt que "convergence vs diversité" pour
la majorité des niches -- le résultat null reste valide et informatif, mais
avec cette réserve.

**Confirmation (5 seeds, avec instrumentation de la saturation)** : pour
lever deux limites identifiées lors de la revue -- (1) avec seulement 3
seeds, le plancher exact du test de Mann-Whitney bilatéral est p=0.10,
rendant p<0.05 structurellement inatteignable sur la comparaison
baseline-vs-test, quel que soit l'effet réel ; (2) la réserve ci-dessus
n'était qu'une estimation par simulation, jamais mesurée sur un run réel --
la campagne a été rejouée à 5 seeds (42, 137, 271, 314, 512, plancher
Mann-Whitney à ~0.008, donc atteignable) avec un compteur de saturation par
génération (`_crowding_saturation_stats`, voir
`Solvers/QINSGA3/algorithm.py`) :

| Indicateur | Baseline (ray-closest) | Test (crowding) | Mann-Whitney (5 seeds) |
|---|---|---|---|
| HV ↑ | 0.145438 | 0.162728 | U=9.0, p=0.547619 |
| GD ↓ | 0.614167 | 0.589421 | U=14.0, p=0.841270 |
| IGD ↓ | 0.625595 | 0.611548 | U=14.0, p=0.841270 |
| Diversité chromosome | 0.004485 | 0.004261 | -- |

Saturation mesurée (pooled sur 1500 générations, 5 seeds × 300 gen) :
**3768 / 4313 niches à ≥2 membres entièrement saturées, soit 87.4%** --
conforme à l'estimation par simulation (75-100%), maintenant une mesure
directe plutôt qu'une réserve théorique. Le résultat reste non significatif
à 5 seeds malgré un plancher de test atteignable (p=0.55 contre un plancher
de 0.008) : ce n'est donc pas un manque de puissance statistique qui masque
un effet réel, le résultat null est robuste.

Scripts conservés : `sensitivity/compare_crowding_guides.py`. Logs complets :
`sensitivity/crowding_guides_campaign_log.txt` (3 seeds),
`sensitivity/crowding_guides_5seed_campaign_log.txt` (5 seeds, avec
saturation instrumentée).

### Remède G -- réparation locale post-décodage (2-opt)

Les remèdes A-F touchent tous le mécanisme de guide/rotation ; ce remède
s'attaque directement à la cause racine identifiée par le diagnostic
initial : le décodeur glouton et irrévocable (`_nearest_neighbour`,
`Solvers/NSGA3/decoder.py`), dont l'effet papillon (petite variation de θ →
tournée complètement différente, écart d'objectif disproportionné) est
mesuré dans `sensitivity/test_theta_route_sensitivity_weighted.py`. Une
première tentative de réparation locale 2-opt existait tout au début de ce
projet ("Premières tentatives" ci-dessus) mais avait été abandonnée --
mauvaise métrique (distance brute au lieu du coût réel avec pénalités de
fenêtres de temps). Ce remède reprend l'idée correctement : recherche
locale 2-opt par tournée, appliquée à chaque génération sur parents ET
enfants, n'acceptant un échange que s'il réduit strictement f1 (coût réel,
pas la distance) et ne dégrade jamais le temps de trajet de la période
au-delà de sa valeur d'avant réparation (garde-fou C13). Réparation
"baldwinienne" : améliore uniquement la fitness évaluée, jamais ré-encodée
dans le chromosome (l'ordre de visite réparé n'a pas d'inverse défini vers
les gènes de priorité). Design complet :
`docs/superpowers/specs/2026-08-04-qinsga3-route-repair-design.md`.
Conséquence directe de ce caractère baldwinien : les valeurs `pareto_F` du
front retourné ne sont reproductibles qu'en ré-exécutant la réparation sur
les chromosomes `pareto_X` retournés, pas par un simple re-décodage --
quiconque ré-analyse plus tard les chromosomes du front de cette campagne
doit le savoir.

**Correction de performance en cours de campagne** : la première
implémentation recalculait f1 sur tout le réseau de tournées à chaque
candidat 2-opt testé, bien au-delà du seuil de 10x du design -- une
première tentative pour compenser (`_MAX_REPAIR_ITER` abaissé de 20 à 5)
n'a quasiment rien changé au ratio (`sensitivity/route_repair_timing_check_iter5.txt` :
instance 100, gen=10 -- baseline=18.9s, réparation=814.3s, **~43x**),
confirmant que le nombre d'itérations n'était pas le facteur dominant.
Remplacé par un calcul de delta local à la seule tournée modifiée
(`_route_f1_contribution`) -- mathématiquement équivalent (tous les
autres termes de f1 sont inchangés par un échange sur une seule tournée
et s'annulent exactement dans la différence, prouvé par un test comparant
au vrai `compute_f1` sur un `route_result` avec une seconde tournée non
touchée), mais O(longueur de tournée) au lieu de O(réseau entier) par
candidat -- `_MAX_REPAIR_ITER` restauré à 20 une fois le vrai goulot
d'étranglement corrigé. Ramène le ratio à **~5.7x** au même test de timing
réduit (`sensitivity/route_repair_timing_check_delta.txt` : instance 100,
gen=10 -- baseline=19.3s, réparation=109.4s) -- mais **~15.2x à pleine
échelle** (voir ci-dessous), le ratio croissant avec l'échelle plutôt que
rester constant.

**Résultat (3 seeds, 300 générations, instance 100 clients,
`sensitivity/compare_route_repair.py`)** :

| Indicateur | Baseline (sans réparation) | Test (réparation) | Mann-Whitney |
|---|---|---|---|
| HV ↑ | 0.125800 | 0.191693 | U=0.0, p=0.100000 |
| GD ↓ | 0.672893 | 0.487900 | U=9.0, p=0.100000 |
| IGD ↓ | 0.671085 | 0.561825 | U=9.0, p=0.100000 |
| Diversité chromosome | 0.003984 | 0.003797 | -- |

**Le signal le plus net des sept remèdes** : sur HV, GD et IGD, les 3 seeds
réparés battent systématiquement les 3 seeds baseline sans exception
(séparation totale, U=0.0/9.0/9.0 sur les 3 métriques) -- HV progresse de
+52 % en relatif (0.1258 → 0.1917). C'est une séparation complète jamais
observée sur les remèdes A-F. p=0.10 reste cependant non significatif :
c'est exactement le plancher exact du test de Mann-Whitney bilatéral à 3
seeds (2/C(6,3) = 0.10, même limite structurelle déjà rencontrée pour le
remède F), pas une absence d'effet -- une séparation totale à 3 seeds est
la configuration la plus favorable possible avant de scaler. Le fossé avec
NSGA-III reste néanmoins significatif (HV p=0.0079, GD p=0.026, IGD
p=0.0045) : la réparation aide mais ne comble pas l'écart. Cette baisse de
diversité chromosome combinée à une hausse de qualité est l'exact inverse
du pattern des remèdes A et B (diversité forcée à la hausse, qualité
dégradée) -- preuve directe, dans les deux sens, que le facteur limitant
n'est pas un déficit de diversité mais la fragilité du décodeur. Sur des
solutions décodées réelles, f2 (CO2) et f3 (temps de trajet) ont aussi été
observés s'améliorer aux côtés de f1 dans tous les cas vérifiés, jamais de
régression -- l'hypothèse de corrélation avec f1 (piloté par la distance)
se vérifie en pratique, même si ce n'est pas suivi dans les métriques
HV/GD/IGD/Spacing du tableau.

**Disqualifié en pratique malgré le signal, sur le coût de calcul** :
même après la correction de performance, le rapport temps réparation/
baseline mesuré à pleine échelle est de **3343.1s / 219.8s ≈ 15.2x** --
plus lent que NSGA-III lui-même, l'algorithme que ce remède essaie de
battre. Décision du porteur de projet : ne pas scaler à 5 seeds pour
confirmer statistiquement le signal -- même confirmé, un remède 15x plus
lent que la référence qu'il compare n'est pas utilisable en pratique.
Documenté tel quel, signal positif mais non actionnable, plutôt que rejeté
pour absence d'effet comme A-F.

Script conservé : `sensitivity/compare_route_repair.py`. Logs complets :
`sensitivity/route_repair_campaign_log.txt` (campagne 3 seeds pleine
échelle), `sensitivity/route_repair_timing_check_iter5.txt` (test de
timing avec `_MAX_REPAIR_ITER=5`, avant la correction de performance),
`sensitivity/route_repair_timing_check_delta.txt` (test de timing après
la correction de performance).

### Remède G — variante pratique : réparation du front final uniquement

La variante ci-dessus (`use_route_repair`) répare chaque individu à chaque
génération -- c'est ce qui la rend ~15x plus lente que le baseline, un coût
structurel, pas un défaut d'implémentation (confirmé après deux
optimisations supplémentaires : fusion des calculs par candidat en un seul
passage sur la tournée, et une recherche 2-opt bornée à une fenêtre de
positions proches -- `sensitivity/route_repair_timing_check_lever1.txt` et
`_timing_check_window.txt`, ratio ramené de 15.2x à ~3.5x à petite échelle,
toujours loin d'être compétitif face à NSGA-III).

Nouveau paramètre `repair_final_front` (`Solvers/QINSGA3/algorithm.py`) :
au lieu de réparer 400 individus × 300 générations, on répare **uniquement
le front de Pareto final retourné, une seule fois**, après la fin de la
recherche évolutive -- la boucle de génération elle-même n'est jamais
ralentie. Même mécanisme de réparation (2-opt baldwinien, évalué sur le
vrai coût f1), juste appliqué à un moment différent du pipeline. Ne teste
plus l'hypothèse scientifique initiale (est-ce que réparer corrige le
signal qui guide toute la recherche) -- répond à une question pratique
différente : peut-on améliorer le résultat rapporté sans ralentir
l'algorithme ?

**Résultat (7 seeds, 300 générations, instance 100 clients,
`sensitivity/compare_route_repair_final.py`)** :

| Indicateur | Baseline (sans réparation) | Test (réparation front final) | Mann-Whitney |
|---|---|---|---|
| HV ↑ | 0.143543 | **0.199913** (+39.3%) | U=7.0, **p=0.026224** |
| GD ↓ | 0.610255 | 0.496704 (-18.6%) | U=38.0, p=0.097319 |
| IGD ↓ | 0.635616 | **0.554527** (-12.8%) | U=42.0, **p=0.026224** |
| Diversité chromosome | 0.004515 | 0.004515 (identique -- le chromosome n'est jamais modifié) | -- |
| Temps moyen | 219.8s | **211.4s** | -- |

**Premier remède avec un gain statistiquement significatif sur cette
instance** (HV et IGD, p<0.05 -- pas juste une séparation totale limitée
par le plancher du test comme pour les remèdes F et G/variante A-F, mais
une vraie significativité, rendue possible par les 7 seeds), **et sans
coût de temps mesurable** (211.4s vs 219.8s, dans le bruit de mesure).
Toujours significativement moins bon que NSGA-III (HV p=0.000068, GD
p=0.000554, IGD p=0.000101) -- l'écart n'est pas comblé, mais c'est la
première fois qu'un remède progresse de façon rigoureusement mesurée, à
coût nul.

Script conservé : `sensitivity/compare_route_repair_final.py`. Log complet :
`sensitivity/route_repair_final_campaign_log.txt`.

### Remède G -- perfectionnements ultérieurs (une piste adoptée, deux rejetées)

Trois pistes indépendantes pour aller plus loin que `repair_final_front`,
explorées après les remèdes H et I (voir plus bas) une fois établi que le
décodeur, pas la diversité, est le vrai levier :

**Rejeté -- fenêtre 2-opt exhaustive** (`sensitivity/
compare_repair_thoroughness.py`) : `_repair_route_result` borne sa
recherche 2-opt à une fenêtre de 8 positions (`_TWO_OPT_WINDOW`), un
compromis vitesse/exhaustivité documenté comme non garanti optimal. Comme
`repair_final_front` ne répare qu'une fois le front final (41-65 solutions),
une fenêtre illimitée reste quasi gratuite (~1.1x-1.8x de candidats en
plus) -- testée en comparaison **appariée** (même front de départ par
seed, seule la réparation change) sur 5 seeds : résultat mitigé, GD pire
sur 5/5 seeds, HV pire sur 4/5. Diagnostic : `_repair_route_result` est en
**première-amélioration** (accepte le premier échange qui améliore) --
élargir la fenêtre change seulement QUEL échange est trouvé en premier,
pas si l'optimum local atteint est meilleur ; la recherche peut partir sur
une trajectoire différente, pas forcément meilleure. Log :
`sensitivity/repair_thoroughness_5seeds_screening_log.txt`.

**Rejeté -- réparation partielle de l'archive en cours de recherche**
(`sensitivity/compare_archive_repair.py`) : au lieu de réparer seulement le
front final une fois, réparer le front de Pareto de la génération courante
(parent ET enfants, ~10-20 individus sur 200, mesuré directement) à chaque
génération, avant qu'il entre dans l'archive -- pour que
`_supplement_from_archive` (déjà en production) voie des guides de niche
réparés pendant la recherche, pas seulement à la fin. Design complet :
`docs/superpowers/specs/2026-08-06-qinsga3-partial-archive-repair-design.md`.
Signal prometteur à 3 seeds (HV +16.8%, GD -10.6%, IGD -8%, les 4 métriques
dans le bon sens) qui s'est **entièrement dissous à 5 seeds** (HV +2.1%,
GD +0.6%, IGD +0.7% -- U entre 11 et 14 sur un maximum de 25, quasiment la
valeur attendue sous absence d'effet) : faux positif de petit échantillon,
pas un vrai signal qui s'affaiblit -- contrairement à `repair_final_front`
lui-même, dont le signal s'était renforcé de 3 à 7 seeds. Coût : ×4.8 au
départ, réduit à ×1.46 après avoir corrigé une implémentation qui réparait
séquentiellement dans le processus principal au lieu d'utiliser le pool de
workers déjà disponible (`_worker_eval_repaired`, déjà utilisé par
`use_route_repair`) -- optimisation gardée dans le script, mais le résultat
de qualité reste nul avec ou sans elle (réparation déterministe : mêmes
seeds -> mêmes résultats, seule la vitesse a changé). Logs :
`sensitivity/archive_repair_5seeds_optimized_screening_log.txt`.

**Adopté -- réparation en meilleure-amélioration** (`sensitivity/
compare_repair_best_improvement.py`) : plutôt qu'élargir la fenêtre
(rejeté ci-dessus), changer la RÈGLE D'ACCEPTATION elle-même -- à chaque
itération, évaluer tous les candidats de la fenêtre (toujours 8, inchangée)
et appliquer strictement le meilleur, pas le premier qui améliore. Garantie
mathématique, pas seulement empirique : à chaque étape, meilleure-
amélioration voit le même candidat que première-amélioration aurait choisi,
plus tous les autres, donc ne peut jamais faire pire à cette étape.
Comparaison **appariée** (même front de départ par seed, recherche
identique, seule la réparation finale change) :

| Indicateur | Victoires (20 seeds) | Wilcoxon signed-rank apparié |
|---|---|---|
| HV ↑ | 18/20 | **p=0.000508** |
| GD ↓ | 19/20 | **p=0.000002** |
| IGD ↓ | 17/20 | **p=0.00003** |
| Spacing ↓ | 9/20 | p=0.74 (non significatif) |

Coût nul (317.7s vs 318.3s en moyenne sur 20 seeds). Face à NSGA-III,
l'écart reste significatif sur HV/GD/IGD (p<0.000001, cohérent avec tous
les remèdes précédents) -- **mais QI-NSGA-III devient significativement
meilleur que NSGA-III sur Spacing** (0.039840 contre 0.054238, p=0.0439) :
un front moins étendu (HV plus faible) mais plus régulièrement réparti,
nouveau et jamais observé aussi nettement dans les remèdes précédents.
Confirmé à 7 seeds (p=0.0156 sur les 3 métriques) avant de scaler à 20 --
signal qui se **renforce** en ajoutant des seeds (p passe de 0.016 à
0.0005/0.000002/0.00003), le pattern inverse du faux positif de la
réparation d'archive ci-dessus, et le signe d'un effet réel. **Adopté en
production** (`Solvers/QINSGA3/repair.py::_repair_route_result`, et
répercuté dans `Livrables_Prof/IRP_100clients_NSGA3_vs_QINSGA3_Colab.ipynb`)
le 2026-08-08. Logs : `sensitivity/repair_best_improvement_20seeds_screening_log.txt`,
`sensitivity/repair_best_improvement_7seeds_vs_nsga3_log.txt`,
`sensitivity/repair_best_improvement_20seeds_vs_nsga3_log.txt`.

### Remède J -- décodeur à anticipation déterministe (non retenu)

Première tentative de cette campagne à changer le PRINCIPE de construction
du décodeur glouton lui-même, plutôt que de le réparer après coup. Une
première tentative dans ce sens existait déjà et avait échoué : le
"décodeur adouci" (`sensitivity/compare_soft_decoder.py`, tirage
aléatoire pondéré au lieu du strict meilleur candidat) -- tendance
négative, probablement à cause du bruit d'évaluation introduit (le même
chromosome peut décoder différemment à chaque appel).

Ce remède teste une alternative **déterministe** : à chaque étape de
construction, au lieu de choisir directement le candidat au meilleur score,
comparer les k=2 meilleurs candidats en simulant un pas de plus pour
chacun (anticipation façon "regret"), et choisir celui qui minimise le
coût cumulé sur 2 pas. Aucun aléatoire introduit -- un même chromosome
décode toujours vers la même tournée. Isolation identique au décodeur
adouci : fork privé du décodeur (`sensitivity/compare_lookahead_decoder.py`),
le décodeur partagé et les résultats NSGA-III (cache) restent inchangés.

**Résultat (3 seeds, 300 générations, instance 100 clients)** :

| Indicateur | Baseline (décodeur original) | Anticipation k=2 | Mann-Whitney |
|---|---|---|---|
| HV ↑ | 0.233612 | 0.147213 (-37%) | **U=9.0, p=0.10 (séparation totale)** |
| GD ↓ | 0.441924 | 0.593290 (+34% pire) | **U=0.0, p=0.10 (séparation totale)** |
| IGD ↓ | 0.515874 | 0.645289 (+25% pire) | **U=0.0, p=0.10 (séparation totale)** |
| Spacing ↓ | 0.044070 | 0.052168 | p=0.40 (non sig.) |

Séparation totale, mais dans le sens **inverse** de celui recherché --
contrairement au décodeur adouci, l'échec n'est pas dû au bruit d'évaluation
(déterministe ici). Hypothèse retenue : changer la règle de construction
gloutonne perturbe la relation implicite que la recherche évolutive avait
apprise entre les gènes de priorité et les tournées résultantes, sans la
remplacer par quelque chose de mieux -- même un choix "plus intelligent"
localement peut casser cet équilibre. Confirme, sur un axe encore différent
(principe de construction, pas réparation post-hoc), que le seul levier qui
fonctionne sur cette instance reste la réparation locale APRÈS décodage
(remède G), pas une modification de la construction elle-même. Non scalé
à 20 seeds (séparation totale déjà dans le mauvais sens à 3 seeds).

Script conservé : `sensitivity/compare_lookahead_decoder.py`. Log complet :
`sensitivity/lookahead_decoder_k2_screening_log.txt`.

### Remède H -- rotation partielle par gène (deux critères, non retenu)

Les remèdes A-G touchent tous la rotation dans son ensemble (fréquence,
cible, règle, ou magnitude) ou le décodeur. Cette piste teste un axe encore
non couvert : au lieu de faire tourner tous les gènes de tous les individus
chaque génération, n'en faire tourner qu'une fraction (`frac=0.15`),
sélectionnée par gène plutôt qu'au hasard par individu -- contrairement à
`rotation_prob` (voir "Premières tentatives" ci-dessus, sans effet mesurable
sur la diversité). Deux critères de sélection testés indépendamment,
`sensitivity/compare_partial_rotation.py` et `sensitivity/
compare_partial_rotation_correlation.py`, design complet dans
`docs/superpowers/specs/2026-08-06-qinsga3-partial-rotation-by-stability-design.md` :

- **H1 -- stabilité** : par individu, on ne fait tourner que les gènes déjà
  les plus proches des élites de la niche (`_elite_rms_distance`, réutilisée
  telle quelle depuis le remède E) -- les gènes divergents restent gelés ce
  tour-ci, façonnés uniquement par SBX/PM.
- **H2 -- corrélation aux améliorations passées** : globalement à la
  population (pas par individu -- une histoire par individu serait rompue
  par le remélange complet qu'effectue la survie élitiste chaque génération,
  le même problème d'identité déjà documenté pour `pbest`/`lambda`), on
  suit une corrélation glissante (fenêtre de 30 générations) entre
  l'amplitude du pas de rotation de chaque gène et la qualité du front ; ne
  sont tournés que les gènes dont l'activité a le mieux suivi les
  améliorations passées.

**Résultat (3 seeds, 300 générations, instance 100 clients)** :

| Indicateur | Baseline | H1 (stabilité) | H2 (corrélation) |
|---|---|---|---|
| HV ↑ | 0.213276 / 0.213909 | 0.172665 (-19 %), U=7.0 p=0.40 | 0.135927 (-36 %), **U=9.0 p=0.10** |
| GD ↓ | 0.480850 / 0.476642 | 0.602452 (+25 %), U=1.0 p=0.20 | 0.680655 (+43 %), **U=0.0 p=0.10** |
| IGD ↓ | 0.543743 / 0.543711 | 0.602943 (+11 %), U=2.0 p=0.40 | 0.648593 (+19 %), **U=0.0 p=0.10** |
| Diversité chromosome | 0.003984 | **0.020392 (×5.1)** | **0.011978 (×3.0)** |

Les deux critères déplacent la diversité chromosome bien plus que tout
remède précédent (H1 : le plus gros mouvement de diversité des huit remèdes
testés), confirmant que cibler QUELS gènes tournent -- et pas seulement
quand ou combien -- change vraiment la dynamique. Mais dans les deux cas la
qualité se dégrade sur HV/GD/IGD, pas seulement sans effet comme
`rotation_prob` : H2 atteint même une séparation totale (U=0.0/9.0, le
plancher p=0.10 du test à 3 seeds, comme les remèdes F et G) mais dans le
sens **inverse** de celui recherché. Même schéma « diversité forcée à la
hausse, qualité dégradée » que les remèdes A et B, sur un axe (par gène)
jamais testé auparavant -- **non scalé à 20 seeds**, même règle de décision
que pour A/B (scaler seulement sur signal positif).

**H1b -- critère inversé (`select=least_stable`)** : hypothèse de suivi --
puisque H1 protège les gènes qui n'ont besoin d'aucune correction (déjà
proches des élites) et abandonne à SBX/PM les gènes qui en auraient le plus
besoin, inverser le critère (ne faire tourner QUE les gènes les plus
éloignés des élites, `sensitivity/compare_partial_rotation.py --select
least_stable`) devrait cibler l'effort de rotation là où il sert vraiment.
Résultat (3 seeds, 300 générations) : encore pire que H1, avec séparation
totale sur 3 métriques sur 4 :

| Indicateur | Baseline | H1b (least_stable) |
|---|---|---|
| HV ↑ | 0.231079 | 0.128473 (-44 %), **U=9.0 p=0.10** |
| GD ↓ | 0.449558 | 0.693727 (+54 % pire), **U=0.0 p=0.10** |
| IGD ↓ | 0.518562 | 0.669857 (+29 % pire), **U=0.0 p=0.10** |
| Diversité chromosome | 0.003984 | 0.004168 (quasi inchangée, +4.6 %) | 

Fait notable : cette fois la diversité chromosome bouge à peine, alors que
c'est le pire résultat de qualité des trois critères -- la preuve la plus
nette que ce n'est pas QUEL sous-ensemble de gènes est choisi qui compte,
mais la **couverture** : limiter la rotation à 15 % des gènes par génération
prive 85 % du génome de correction dirigée à chaque génération pendant tout
le run, quel que soit le critère de sélection.

Scripts conservés : `sensitivity/compare_partial_rotation.py` (`--select
most_stable|least_stable`), `sensitivity/
compare_partial_rotation_correlation.py`. Logs complets :
`sensitivity/partial_rotation_screening_log.txt`,
`sensitivity/partial_rotation_correlation_screening_log.txt`,
`sensitivity/partial_rotation_least_stable_screening_log.txt`.

### Remède I -- archive comportementale (diversité structurelle des tournées, non retenu)

Piste distincte des remèdes A-H : au lieu de toucher la dynamique de
recherche (rotation), on cible l'élagage final de l'archive (`_crowding_trim`,
`algorithm.py:964-974`), qui aujourd'hui sélectionne les solutions à
rapporter uniquement par crowding distance en espace objectif (F) --
X/theta ne sont que des données transportées. Hypothèse : deux tournées
structurellement très différentes peuvent avoir un coût quasi identique (les
quatre objectifs sont des sommes sur les arcs/flux, `Solvers/NSGA3/
evaluator.py`), donc l'élagage F-seul peut retenir des tournées redondantes
et jeter des tournées structurellement distinctes.

Contexte du diagnostic déjà fait (H3, section "Diagnostic diversité/
décodeur") : la diversité des tournées décodées de la population entière
n'est que légèrement inférieure à celle de NSGA-III (ratio 0.846, très loin
du ratio chromosome ×13) -- donc la prémisse "les tournées sont redondantes"
n'est pas fortement soutenue au niveau population. Ce remède teste une
question plus étroite : l'archive, elle, quand elle élague, jette-t-elle
quand même de la diversité structurelle pour rien ?

**Calibrage nécessaire** : l'élagage final ne se déclenche que si le front
dépasse `pop_size` (200) -- or les fronts finaux mesurés font 41 à 65
solutions, donc l'élagage à `pop_size` ne s'active quasiment jamais. Testé
à la place à une taille de rapport plus petite (`report_size=30`, un besoin
réaliste : présenter un sous-ensemble diversifié gérable à un décideur),
appliquée identiquement au baseline (F-crowding seul) et au test (F-crowding
+ diversité structurelle de Jaccard, réutilisant `_route_arcset`/
`_jaccard_distance` de `sensitivity/test_route_diversity.py`), score combiné
`w_structural * structurel + (1-w_structural) * F`. Contrairement aux
remèdes A-H, celui-ci ne touche jamais la boucle de recherche -- seulement
quels non-dominés (déjà tous Pareto-valides) sont rapportés à la fin.

**Résultat (3 seeds, 300 générations, deux poids testés)** :

| Indicateur | Baseline | w=0.5 | w=0.2 |
|---|---|---|---|
| HV ↑ | 0.181749 | 0.121062 (-33 %), p=0.40 | 0.123295 (-32 %), p=0.40 |
| GD ↓ | 0.572265 | 0.704934 (+23 % pire), p=0.20 | 0.703159 (+23 % pire), p=0.20 |
| IGD ↓ | 0.585119 | 0.680763 (+16 % pire), p=0.40 | 0.676687 (+16 % pire), p=0.40 |
| Diversité tournées (cible) | 0.410363 | 0.434851 (+6 %), p=0.40 | 0.424997 (+3.6 %), p=1.00 |

Le compromis n'est favorable à aucun des deux poids : réduire `w_structural`
de 0.5 à 0.2 ne réduit presque pas le coût en qualité (HV toujours -32/33 %)
mais réduit le gain sur la métrique visée -- le pire des deux mondes. Le
critère F-crowding de production semble déjà proche d'un optimum pour ce
problème : même une déviation modeste vers la diversité structurelle coûte
cher en HV/GD/IGD (mesurées entièrement en espace F) sans acheter grand-chose
en diversité de tournées, ce qui suggère que les points extrêmes du front
(ceux qui maximisent le spread F) sont déjà de bons représentants de la
diversité structurelle. Non scalé à 20 seeds (même règle : signal négatif
aux deux poids testés).

Script conservé : `sensitivity/compare_behavioral_archive.py`. Logs
complets : `sensitivity/behavioral_archive_screening_log.txt`,
`sensitivity/behavioral_archive_w02_screening_log.txt`.

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
  limite QI-NSGA-III sur l'IRP -- confirmé à 5 seeds (p=0.55, toujours non
  significatif malgré un plancher de test atteignable) et avec la réserve,
  mesurée directement et non plus estimée (87.4% des niches à ≥2 membres
  entièrement saturées sur 1500 générations), que la crowding distance
  sature en pratique pour la grande majorité des niches sur l'IRP (voir
  section Remède F).
- Le remède G, seul remède à s'attaquer au décodeur plutôt qu'au mécanisme
  de guide/rotation (réparation locale 2-opt post-décodage, évaluée sur le
  vrai coût f1), montre le signal de qualité le plus net des sept remèdes :
  séparation totale sur HV/GD/IGD (les 3 seeds réparés battent
  systématiquement les 3 seeds baseline, HV +52 % en relatif), p=0.10
  restant le plancher structurel du test à 3 seeds, pas une absence
  d'effet. Mais le coût de calcul (~15x plus lent que le baseline à pleine
  échelle, plus lent que NSGA-III lui-même) le disqualifie en pratique
  indépendamment du résultat statistique -- documenté comme signal positif
  non actionnable plutôt que rejeté comme A-F. Sa variante pratique
  (`repair_final_front` -- réparer uniquement le front de Pareto final,
  une fois, au lieu de chaque individu à chaque génération) lève cette
  disqualification : même mécanisme de réparation, coût de temps
  quasi-nul (211.4s vs 219.8s baseline), et à 7 seeds, un gain
  **statistiquement significatif** sur HV (+39.3 %, p=0.026) et IGD
  (-12.8 %, p=0.026) -- le premier de tous les remèdes testés (A-G) à
  atteindre une vraie significativité plutôt qu'une séparation totale
  limitée par le plancher du test à 3 seeds. Toujours significativement
  moins bon que NSGA-III.
- Trois perfectionnements du remède G testés ensuite (section "Remède G --
  perfectionnements ultérieurs") : élargir la fenêtre 2-opt (rejeté,
  mitigé/négatif) et réparer l'archive en continu pendant la recherche
  (rejeté, faux positif à 3 seeds qui s'est dissous à 5) échouent tous les
  deux -- mais passer de première- à **meilleure-amélioration** dans la
  réparation elle-même réussit, avec le signal le plus net et le plus
  robuste de toute la campagne (Wilcoxon apparié à 20 seeds : HV p=0.0005,
  GD p=0.000002, IGD p=0.00003, coût nul, signal qui se renforce avec
  l'échantillon au lieu de se diluer) -- **adopté en production**. Révèle
  aussi un résultat nouveau : QI-NSGA-III devient significativement meilleur
  que NSGA-III sur Spacing (p=0.044) à cette échelle, même si HV/GD/IGD
  restent significativement en sa défaveur.
- Le remède J, première tentative à changer le PRINCIPE de construction du
  décodeur (anticipation déterministe à 2 pas) plutôt que de le réparer
  après coup, échoue aussi -- séparation totale à 3 seeds mais dans le
  mauvais sens (HV -37 %, GD +34 %, IGD +25 %), sans le défaut du décodeur
  adouci (bruit d'évaluation, puisque déterministe) : changer la règle de
  construction perturbe la relation apprise entre gènes de priorité et
  tournées, sans la remplacer par mieux. Dixième confirmation indépendante
  que seule la réparation locale POST-décodage (remède G) fonctionne sur
  cette instance -- ni la diversité, ni la construction elle-même.
- Le remède H, qui cible pour la première fois QUELS gènes tournent (par
  stabilité ou par corrélation aux améliorations passées) plutôt que la
  fréquence, la cible ou la règle de la rotation dans son ensemble, produit
  le plus gros déplacement de diversité chromosome de tous les remèdes
  (×5.1 et ×3.0) -- confirmant que l'axe « par gène » a un effet réel,
  contrairement à `rotation_prob` (par individu, aléatoire) qui n'avait
  quasiment rien bougé. Mais la qualité se dégrade dans les deux variantes,
  avec pour la variante par corrélation une séparation totale (p=0.10) dans
  le sens inverse de celui recherché -- huitième confirmation indépendante
  que plus de diversité chromosome ne comble pas l'écart avec NSGA-III sur
  cette instance. Une troisième variante (H1b, critère de stabilité inversé
  -- faire tourner les gènes les MOINS stables, ceux qui ont le plus besoin
  d'être corrigés) donne le résultat le pire des trois (séparation totale
  sur HV/GD/IGD) alors que la diversité chromosome, cette fois, bouge à
  peine -- la preuve la plus nette que ce n'est pas QUEL sous-ensemble de
  gènes est choisi qui compte, mais que limiter la couverture de la rotation
  à 15 % du génome, quel que soit le critère, prive le reste d'une
  correction dirigée pendant tout le run.
- Le remède I, seul remède à toucher l'élagage de l'archive plutôt que la
  dynamique de recherche (aucun risque sur la convergence -- il ne choisit
  qu'entre solutions déjà non-dominées), teste si combiner le crowding en
  espace F avec une diversité structurelle des tournées (distance de Jaccard
  sur les arcs) améliore le front rapporté. Aux deux poids testés (0.5 et
  0.2), le gain sur la diversité de tournées reste faible et non significatif
  (+6 % puis +3.6 %) pour un coût constant en qualité (HV -32/33 %, non
  significatif à 3 seeds mais cohérent) -- réduire le poids ne réduit pas le
  coût, seulement le bénéfice. Neuvième confirmation indépendante, sur un
  axe pourtant différent (représentation du front final, pas dynamique de
  recherche) : le critère F-crowding de production est déjà difficile à
  battre pour ce problème.

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
