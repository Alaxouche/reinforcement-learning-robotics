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
│   │       ├── bras_robotique_2d.py   # Mécanique commune
│   │       ├── bras_robot_naive.py    # +10 / -0.02
│   │       ├── bras_robot_smart.py    # récompense à potentiel, bonus terminal +10
│   │       └── bras_robot_distance.py # d(k) - d(k+1) + 10 si cible atteinte
│   │
│   └── visualisation/
│       └── graphes.py                 # Fonctions de tracé
│
├── experiences/
│   ├── entrainement.py
│   ├── entrainement_bras_robotique.py
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
bras, puis enregistre les graphiques PNG (≈ 2 minutes) :

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

**Deux graphiques des récompenses du bras** :
- taux de réussite de la politique tous les 500 épisodes (mêmes 30 départs) ;
- longueur des épisodes pendant l'entraînement (moyenne glissante).

Le graphique de comparaison labyrinthe/bras a été retiré.
Le graphe 4 affiche des **flèches et un chemin dans le labyrinthe**.
Pour le bras, il présente **quatre couleurs selon l'action préférée**
(theta1+, theta1-, theta2+, theta2-), sans texte ni chemin superposé ;
les arrivées sont entourées de cercles.

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
- **Smart** : une récompense de base `+10` à l'arrivée et `-0.02` sinon,
  complétée par un **guidage à potentiel de distance** :
  `r = r_base + distance_avant - 0.95 * distance_apres` hors arrivée,
  `r = 10 + distance_avant` à l'arrivée (potentiel terminal nul).
  Le coefficient `0.95` correspond au facteur `gamma` du Q-learning et
  doit rester égal à celui-ci si les hyperparamètres changent.
- **Distance** : `distance_avant - distance_apres + (10 si termine sinon 0)`.
  On évalue le rapprochement pendant le mouvement avec le même bonus terminal.

Les trois modes comprennent un **bonus terminal de +10**. Les modes naive
et smart utilisent `-0.02` par mouvement non terminal ; smart ajoute un
signal de proximité fondé sur la différence des potentiels de distance.
Le mode distance utilise directement le changement de distance sans pénalité fixe.

Les trois variantes utilisent les mêmes états, actions, angles, cible
et hyperparamètres. Seule la fonction de récompense change.
Aucune des trois ne garantit à elle seule une politique optimale.
Les récompenses numériques ne sont pas directement comparables :
on évalue plutôt la proportion de réussites et la distance finale.

Il s'agit d'un modele cinematique simplifie :
il n'inclut pas la dynamique des moteurs ni les collisions pendant
la rotation continue entre deux poses. Aucune butee articulaire
n'est imposee dans cette simulation.

### Tests automatiques du bras

Depuis la racine du dépôt :

```bash
py -m unittest discover -s tests -v
```

Les tests vérifient les trois fonctions de récompense, les départs
aléatoires reproductibles et un petit entraînement sans affichage graphique.
`tests/test_graphes.py` vérifie les compteurs utilisés par les graphes, et que
les graphes ne modifient pas l'entraînement.

### Lancer les trois modes du bras

Les trois entraînements (naive, smart, distance) sont exécutés
automatiquement l'un après l'autre :

```bash
py -m experiences.entrainement_bras_robotique
```

On peut aussi choisir un mode et la seed :

```bash
py -m experiences.entrainement_bras_robotique --mode naive
py -m experiences.entrainement_bras_robotique --mode smart --seed 42
py -m experiences.entrainement_bras_robotique --mode distance --seed 42
```

Par défaut, `--seed 42`.
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
