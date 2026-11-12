# Projet RL — Q-learning sur environnements discrets

Le projet généralise les environnements discrets tout en gardant un seul algorithme : le Q-learning.

## Architecture

```text
reinforcement-learning-robotics/
├── src/
│   ├── algorithmes/
│   │   └── q_learning.py
│   │
│   ├── environnements/
│   │   ├── environnement_discret.py
│   │   ├── labyrinthe/
│   │   │   ├── labyrinthe.py
│   │   │   ├── labyrinthe_5x5.py
│   │   │   └── labyrinthe_36x36.py    # grille ouverte générée, 60 pièges
│   │   └── bras_robotique/
│   │       ├── bras_robotique_2d.py   # Mécanique commune (4 ou 9 actions)
│   │       ├── bras_robot_naive.py    # +10 / -0.02
│   │       ├── bras_robot_smart.py    # -d(k+1) + 10 si cible atteinte
│   │       └── bras_robot_distance.py # d(k) - d(k+1) + 10 si cible atteinte
│   │
│   └── visualisation/
│       └── graphes.py                 # Fonctions de tracé
│
├── experiences/
│   ├── entrainement.py
│   ├── entrainement_bras_robotique.py
│   ├── etude_epsilon.py               # Impact de epsilon, Q initiale constante / aléatoire
│   ├── graphes.py                     # Génère tous les graphes
│   └── README_graphes.md
│
├── tests/
│   ├── test_bras_robotique.py
│   └── test_graphes.py
│
├── requirements.txt
└── README.md
```

Cette organisation reste volontairement simple :

- `src/algorithmes/` contient l'algorithme Q-learning ;
- `src/environnements/` contient les classes liées aux environnements ;
- `src/visualisation/` contient les fonctions de tracé ;
- `experiences/` contient les scripts qui assemblent un environnement et l'algorithme.

## Héritage des environnements

```text
EnvironnementDiscret
        ↑
    Labyrinthe
        ↑
  Labyrinthe5x5, Labyrinthe36x36
```

`EnvironnementDiscret` définit le contrat commun : nombre d'états, nombre d'actions, `reset()`, `actions_possibles()` et `step()`.

`Labyrinthe` contient la mécanique commune aux labyrinthes : déplacements, limites de la grille, murs, calcul de l'état suivant, `reset()` et `step()`. La fonction `recompense(etat)` reste abstraite.

`Labyrinthe5x5` contient la configuration concrète :

```text
 0   1   2   3   4
 5   6   7   8   9
10  11  12  13  14
15  16  17  18  19
20  21  22  23  24
```

```text
départ   : 0
arrivée  : 24
murs     : 3, 11, 13
pièges   : 7, 17
```

Récompenses :

```text
case normale : -0.1
piège        : -1.0
arrivée      : +1.0
```

Les actions sont :

```text
0 = Haut
1 = Bas
2 = Gauche
3 = Droite
```

### Labyrinthe 36 x 36

`Labyrinthe36x36` sert à comparer le Q-learning sur un problème beaucoup plus
grand (1 296 états contre 25). La grille est **ouverte** : des segments de murs
droits sont posés au hasard (≈ 15 % des cases, graine fixe `seed=0`, donc
toujours le même labyrinthe) et 60 pièges sont dispersés. Il existe donc de
très nombreux chemins viables : l'agent doit trouver celui qui maximise le
retour (un piège coûte autant que 10 déplacements).

```text
départ : 0 (coin haut gauche)    arrivée : 1295 (coin bas droit)
murs   : 197                     pièges  : 60
plus court chemin : 70 déplacements (retour optimal -5.9, sans piège)
récompenses : identiques au 5 x 5
```

Les hyperparamètres sont adaptés à la taille :

| | 5 x 5 | 36 x 36 |
|---|---|---|
| alpha | 0.1 | 0.5 |
| gamma | 0.9 | 0.99 |
| décroissance d'epsilon | 0.99 | 0.998 |
| épisodes | 1 000 | 3 000 |
| pas max par épisode | 30 | 1 000 |

Avec alpha = 0.1 le 36 x 36 n'apprenait pas de chemin vers l'arrivée en
3 000 épisodes ; gamma = 0.99 donne un horizon compatible avec un trajet
de 70 pas.

Résultat (graine 42) : depuis le départ, l'agent suit un chemin optimal de
70 déplacements dans les deux labyrinthes. Mais depuis une case quelconque, la
politique du 36 x 36 n'est optimale que pour ≈ 65 % des cases (100 % dans
le 5 x 5) : l'agent part toujours de S, et les zones éloignées de son trajet,
peu visitées une fois epsilon faible, restent mal apprises.

## Q-learning

`QLearning` reste une classe autonome et utilise un `EnvironnementDiscret`.

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
```

## Exécution

Depuis la racine du projet :

```bash
py -m experiences.entrainement              # 5 x 5 puis 36 x 36, avec comparaison
py -m experiences.entrainement --taille 36  # un seul labyrinthe
```

## Graphiques

Les scripts d'entraînement affichent leurs résultats dans le terminal, sans produire
d'image. Un seul programme entraîne les deux labyrinthes et les trois modes du
bras, à 4 puis à 9 actions, puis enregistre les graphiques PNG (≈ 2 minutes) :

```bash
py -m pip install -r requirements.txt
py -m experiences.graphes
```

Pour chaque environnement : Q-values brutes (3), meilleure action (4),
visites (6, avec dans chaque case passages / mises à jour de Q) et
passages par type de déplacement (6b, une carte par action, même échelle
bleue). Pour les labyrinthes, en plus : écart à la politique optimale depuis
chaque case (7), et une **comparaison 5 x 5 / 36 x 36**
(`labyrinthes/comparaison_5x5_36x36.png`).

**Deux graphiques des récompenses du bras**, pour 4 et pour 9 actions
(`bras_robotique/4_actions/` et `bras_robotique/9_actions/`) :
- taux de réussite de la politique tous les 500 épisodes (mêmes 30 départs) ;
- longueur des épisodes pendant l'entraînement (moyenne glissante).

Plus une **comparaison 4 / 9 actions** (`bras_robotique/comparaison_4_9_actions.png`).
L'étude de epsilon a son propre programme (voir
[Impact de epsilon sur l'exploration](#impact-de-epsilon-sur-lexploration)).

Le graphique de comparaison labyrinthe/bras a été retiré.
Le graphe 4 affiche des **flèches et un chemin dans le labyrinthe**.
Pour le bras à 4 actions, il présente **quatre couleurs selon l'action préférée**
(theta1+, theta1-, theta2+, theta2-), sans texte ni chemin superposé ;
les arrivées sont entourées de cercles. À 9 actions, neuf couleurs ne se
distinguent plus : la meilleure action est décomposée en **deux cartes**, la
variation choisie pour theta1 et celle pour theta2 (diminue / immobile / augmente).

Les images sont enregistrées dans `experiences/resultats/`.
Pour interpréter les graphiques :
[experiences/README_graphes.md](experiences/README_graphes.md).

## Deuxieme environnement : bras robotique 2D

Le robot comporte deux segments de longueurs L1 et L2 fixes.
Theta1 est mesure par rapport a l'horizontale, theta2 est l'angle
du second segment par rapport au premier. Les coordonnees de
l'effecteur sont :

```text
x = L1*cos(theta1) + L2*cos(theta1+theta2)
y = L1*sin(theta1) + L2*sin(theta1+theta2)
```

Les angles sont discrets et independamment libres de tourner sur 360 degres.
Les quatre actions augmentent ou diminuent l'un des angles de 10 degres
(par defaut), avec bouclage modulo 360 : 350 + 10 = 0 degre.
Avec `nb_actions=9`, les deux articulations bougent **en même temps** : chacune
reçoit -10, 0 ou +10 degrés, soit 3 x 3 = 9 actions (voir
[Passage à 9 actions](#passage-à-9-actions)).
Les angles vont de 0 a 350 degres inclus (360 equivaut a 0).
Cela produit 36 x 36 = 1296 etats, sans contraintes de sol ni de quadrant.
La cible peut avoir des coordonnees x et y negatives.

La cible M0=(x0,y0) est **fixe pendant chaque entraînement**.
Pendant l'entraînement, les deux angles initiaux sont tirés uniformément
parmi les configurations discrètes non terminales à chaque épisode.
La base du bras reste en (0,0), mais son extrémité démarre à des positions
variées. Le départ historique (theta1, theta2)=(90,0) degrés reste utilisé
comme référence d'évaluation, et la cible par défaut est M0=(1,1),
avec une tolérance de 0.15 unité de longueur.
Le simulateur verifie qu'une configuration discretisee peut atteindre
la cible dans la tolerance choisie. La cible peut etre situee dans
n'importe lequel des quatre quadrants, mais la portee physique
reste limitee par les longueurs L1 et L2 : la distance a l'origine
est comprise entre abs(L1-L2) et L1+L2 (avant discretisation).

La mécanique est partagée dans `bras_robotique_2d.py` (géométrie,
actions, transitions, tirages aléatoires). Les récompenses sont définies
dans **trois fichiers distincts**, chacun contenant une classe de bras :

- `bras_robot_naive.py` → `BrasRobotNaive`
- `bras_robot_smart.py` → `BrasRobotSmart`
- `bras_robot_distance.py` → `BrasRobotDistance`

Trois récompenses sont disponibles pour le même simulateur :

- **Naive (parcimonieuse)** : `+10` lorsque la cible est atteinte,
  `-0.02` sinon.
- **Smart** : `-distance_apres + (10 si termine sinon 0)`.
  Le robot est moins pénalisé lorsqu'il termine son mouvement plus près
  de la cible ; à l'arrivée, il reçoit également un bonus de `+10`.
- **Distance** : `distance_avant - distance_apres + (10 si termine sinon 0)`.
  On évalue le rapprochement pendant le mouvement avec le même bonus terminal.

Les trois modes comprennent un **bonus terminal de +10**.
Seul naive applique la pénalité fixe `-0.02` hors arrivée.
Smart pénalise la distance restante, tandis que distance récompense
la variation de distance pendant le mouvement.

Les trois variantes utilisent les mêmes états, actions, angles, cible
et hyperparamètres. Seule la fonction de récompense change.
Aucune des trois ne garantit à elle seule une politique optimale.
Les récompenses numériques ne sont pas directement comparables :
on évalue plutôt la proportion de réussites et la distance finale.

Il s'agit d'un modele cinematique simplifie :
il n'inclut pas la dynamique des moteurs ni les collisions pendant
la rotation continue entre deux poses. Aucune butee articulaire
n'est imposee dans cette simulation.

### Passage à 9 actions

Avec 4 actions, une seule articulation bouge à chaque pas. Avec 9 actions,
chaque articulation reçoit indépendamment -10, 0 ou +10 degrés :

| Action | theta1 | theta2 | Action | theta1 | theta2 |
|---|---|---|---|---|---|
| 0 `theta1 +` | +10 | 0 | 5 `theta1 +, theta2 -` | +10 | -10 |
| 1 `theta1 -` | -10 | 0 | 6 `theta1 -, theta2 +` | -10 | +10 |
| 2 `theta2 +` | 0 | +10 | 7 `theta1 -, theta2 -` | -10 | -10 |
| 3 `theta2 -` | 0 | -10 | 8 `immobile` | 0 | 0 |
| 4 `theta1 +, theta2 +` | +10 | +10 | | | |

Les quatre premières actions sont celles de la version à 4 actions. Le choix
se fait par `nb_actions=9` dans le constructeur des trois classes de bras
(4 par défaut). Les états, les récompenses et les hyperparamètres ne changent
pas ; la Q-table passe de 1296 x 4 à 1296 x 9 (11 646 couples (s, a)
autorisés au lieu de 5 176). L'action `immobile` ne rapproche jamais de la
cible : elle coûte `-0.02` (naive), `-d` (smart) ou `0` (distance).

Résultats (graine 42, 5 000 épisodes, mêmes 30 départs de test, sans
exploration, 50 mouvements maximum) :

| Récompense | 4 actions : réussite | mouvements (réussites) | 9 actions : réussite | mouvements (réussites) |
|---|---|---|---|---|
| naive | 96.7 % | 15.0 | 100 % | 11.5 |
| smart | 66.7 % | 12.6 | 63.3 % | 8.3 |
| distance | 100 % | 17.1 | 100 % | 12.5 |

- Les mouvements simultanés raccourcissent les trajets de 23 à 34 %. Les deux
  articulations bougent ensemble dans 55 à 61 % des états non terminaux
  selon la récompense.
- La table est 2,25 fois plus grande, donc chaque couple (s, a) est mis à
  jour moins souvent (≈ 10 fois au lieu de ≈ 27). L'apprentissage n'est
  pourtant pas plus lent pour naive et distance : les épisodes plus courts
  propagent plus vite le bonus terminal (`comparaison_4_9_actions.png`).
- Smart reste vers 65 % : sa limite vient de la récompense, pas du jeu
  d'actions. L'action `immobile` y devient même un piège : elle est la
  meilleure action apprise dans 51 états (14 pour naive, aucun pour
  distance). Rester sur place coûte `-d` à chaque pas, mais un détour qui
  éloigne d'abord de la cible coûte davantage ; la politique gloutonne peut
  donc s'arrêter loin de la cible.

### Impact de epsilon sur l'exploration

```bash
py -m experiences.etude_epsilon                      # 9 actions, graine 42 (≈ 3 minutes)
py -m experiences.etude_epsilon --seeds 42 43 44     # moyenne sur trois graines
py -m experiences.etude_epsilon --actions 4 --modes naive distance
```

Pour isoler l'effet d'epsilon, il reste **constant** pendant tout
l'entraînement (`epsilon_min = epsilon`, décroissance = 1), pour
epsilon = 0, 0.05, 0.1, 0.2, 0.5 et 1. Deux initialisations de la Q-table :

- **constante** : `QLearning(..., initialisation="constante", valeur_initiale=0.0)`,
  Q(s, a) = 0 partout. En cas d'égalité, `argmax` prend l'action de plus
  petit indice ;
- **aléatoire** (par défaut) : Q(s, a) uniforme dans [-0.1 ; 0.1].

Tout le reste suit la référence du bras (9 actions, alpha = 0.2,
gamma = 0.95, 5 000 épisodes, départs aléatoires). Mesures : part des couples
(s, a) essayés au moins une fois, réussite gloutonne sur les 30 départs de
test, longueur moyenne des 500 derniers épisodes. Graphes :
`experiences/resultats/bras_robotique/epsilon/9_actions/`.

Couples (s, a) essayés en fin d'entraînement (graine 42) :

| epsilon | 0 | 0.05 | 0.1 | 0.2 | 0.5 | 1 |
|---|---|---|---|---|---|---|
| naive, Q constante | 98.1 % | 99.3 % | 99.3 % | 99.7 % | 100 % | 100 % |
| naive, Q aléatoire | 45.3 % | 61.4 % | 70.3 % | 84.3 % | 98.9 % | 100 % |
| smart, Q constante | 98.5 % | 99.4 % | 99.5 % | 99.7 % | 100 % | 100 % |
| smart, Q aléatoire | 97.2 % | 98.3 % | 98.8 % | 99.4 % | 100 % | 100 % |
| distance, Q constante | 17.1 % | 34.0 % | 45.0 % | 60.5 % | 89.0 % | 100 % |
| distance, Q aléatoire | 18.3 % | 36.1 % | 45.8 % | 62.1 % | 91.8 % | 100 % |

Réussite gloutonne (graine 42) :

| epsilon | 0 | 0.05 | 0.1 | 0.2 | 0.5 | 1 |
|---|---|---|---|---|---|---|
| naive, Q constante | 93.3 % | 96.7 % | 100 % | 96.7 % | 100 % | 73.3 % |
| naive, Q aléatoire | 96.7 % | 100 % | 96.7 % | 96.7 % | 100 % | 76.7 % |
| smart, Q constante | 63.3 % | 66.7 % | 63.3 % | 63.3 % | 63.3 % | 46.7 % |
| smart, Q aléatoire | 70.0 % | 63.3 % | 60.0 % | 63.3 % | 70.0 % | 53.3 % |
| distance, Q constante | 80.0 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| distance, Q aléatoire | 53.3 % | 100 % | 100 % | 100 % | 100 % | 100 % |

Interprétation :

1. **L'exploration ne dépend pas seulement d'epsilon**, mais aussi de la
   position de la valeur initiale par rapport aux récompenses.
   Avec naive et smart, toute récompense hors arrivée est négative. Q = 0 est
   donc **optimiste** : une action essayée passe sous 0, et les actions encore
   à 0 gagnent l'argmax. L'agent glouton essaie ainsi systématiquement les
   actions nouvelles : 98 % des couples sont essayés même avec epsilon = 0.
2. Avec la **Q aléatoire** et naive, la pénalité `-0.02` est faible : après
   une mise à jour, Q ≈ -0.004 reste au-dessus d'environ la moitié des
   valeurs initiales tirées dans [-0.1 ; 0.1]. L'optimisme est en partie
   perdu et la couverture suit epsilon (45 % à 100 %). Avec smart, `-d`
   vaut -1.8 en moyenne : les deux initialisations restent optimistes (≥ 97 %).
3. Avec distance, se rapprocher rapporte une récompense **positive** :
   Q = 0 n'est plus optimiste. L'agent garde la première action qui a
   rapproché la cible, et seule epsilon fait essayer les autres. La
   couverture passe de 17-18 % (epsilon = 0) à 100 % (epsilon = 1), quelle que
   soit l'initialisation. Sans exploration, la réussite tombe à 80 %
   (constante) ou 53 % (aléatoire) ; epsilon = 0.05 suffit pour 100 %.
4. **Trop d'exploration coûte cher** : la longueur des épisodes passe
   de 12 à 17 mouvements (epsilon = 0.05 ou 0.1) à 23-25 (0.5) et 49 (1,
   presque toujours la limite de 50). Le Q-learning apprend hors politique, donc une
   politique gloutonne reste apprise à epsilon = 1 (100 % avec distance), mais
   les récompenses naive et smart en souffrent (73-77 % et 47-53 %) : les
   marches aléatoires atteignent rarement la cible, et le bonus terminal se
   propage lentement.
5. Avec Q constante, l'exploration gloutonne optimiste est **plus rapide**
   que le hasard au début : après 250 épisodes (naive et smart), 86-88 % des
   couples sont essayés avec epsilon = 0, contre 62.5 % avec epsilon = 1
   (`couverture_couples.png`). Le hasard rejoue souvent des couples déjà
   essayés.

En pratique, une petite exploration (epsilon ≈ 0.05 à 0.2), ou la décroissance
de 1 vers 0.15 utilisée par les expériences de référence, suffit dès que
l'initialisation n'est pas optimiste.

Les tableaux reposent sur la graine 42. Moyennées sur les graines 42, 43 et
44 (`--seeds 42 43 44`), les couvertures restent à moins de 3 points de ces
valeurs et les cinq constats ci-dessus sont inchangés. Avec distance et
epsilon = 0, la réussite moyenne est de 84 % (Q constante) contre 47 %
(Q aléatoire) ; elle est de 100 % pour toutes les graines dès epsilon = 0.05.

### Tests automatiques du bras

Depuis la racine du dépôt :

```bash
py -m unittest discover -s tests -v
```

Les tests vérifient les trois fonctions de récompense, les départs
aléatoires reproductibles, les 9 actions (mouvements simultanés, bouclage,
action immobile), l'initialisation constante de Q et un petit entraînement
sans affichage graphique.
`tests/test_graphes.py` vérifie les compteurs utilisés par les graphes, et que
les graphes ne modifient pas l'entraînement.

### Lancer les trois modes du bras

Les trois entraînements (naive, smart, distance) sont exécutés
automatiquement l'un après l'autre, à 4 puis à 9 actions :

```bash
py -m experiences.entrainement_bras_robotique
```

On peut aussi choisir un mode, le nombre d'actions et la seed :

```bash
py -m experiences.entrainement_bras_robotique --mode naive
py -m experiences.entrainement_bras_robotique --mode smart --actions 9
py -m experiences.entrainement_bras_robotique --mode distance --actions 4 --seed 42
```

Par défaut, `--seed 42` et `--actions tous` (4 puis 9).
L'entraînement utilise 5 000 épisodes, `epsilon_min=0.15` et une
décroissance de `0.9995` pour conserver davantage d'exploration.
La graine est réappliquée à chaque mode ; le tirage des états de départ
utilise un générateur indépendant du tirage des actions.

Pour comparer les agents, chacun est testé sans exploration sur les
**30 mêmes configurations initiales** tirées avec une graine indépendante.
Le terminal affiche le taux de réussite, les mouvements moyens pour
les essais réussis et la distance finale moyenne. Le départ historique
(90°, 0°) est également testé séparément.
Ces mesures sur une seule graine ne suffisent pas à prouver la convergence
ni à établir qu'un mode est systématiquement supérieur.

Les paramètres (longueurs, pas angulaire, position cible, tolérance et
nombre d'épisodes) sont modifiables dans
`experiences/entrainement_bras_robotique.py`.
Les trois classes dérivent de la base commune `BrasRobotique2D`.
Le départ reste fixe par défaut (`depart_aleatoire=False`) si une
de ces classes est utilisée indépendamment du script d'entraînement.
`reset(etat_depart=...)` permet d'imposer un état pour une évaluation.

A noter : si la cible change entre les episodes d'un meme entrainement,
il faut integrer sa position dans l'etat RL ; cette premiere version
apprend uniquement pour une cible fixe.
