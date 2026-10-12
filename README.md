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
│   │   ├── labyrinthe.py
│   │   ├── labyrinthe_3x3.py
│   │   ├── labyrinthe_5x5.py
│   │   └── labyrinthe_9x9.py
│   │
│   └── visualisation/
│       └── graphes.py
│
├── experiences/
│   ├── entrainement.py
│   └── graphes.py
│
├── resultats/          (généré par experiences.graphes)
│
├── requirements.txt
└── README.md
```

Cette organisation reste volontairement simple :

- `src/algorithmes/` contient l'algorithme Q-learning ;
- `src/environnements/` contient les classes liées aux environnements ;
- `src/visualisation/` contient les fonctions de tracé (matplotlib) ;
- `experiences/` contient les scripts qui assemblent un environnement et l'algorithme pour lancer un entraînement ou générer les graphes.

## Héritage des environnements

```text
EnvironnementDiscret
        ↑
             Labyrinthe
        ↑         ↑         ↑
Labyrinthe3x3 Labyrinthe5x5 Labyrinthe9x9
```

`EnvironnementDiscret` définit le contrat commun : nombre d'états, nombre d'actions, `reset()`, `actions_possibles()` et `step()`.

`Labyrinthe` contient la mécanique commune aux labyrinthes : déplacements, limites de la grille, murs, calcul de l'état suivant, `reset()` et `step()`. La fonction `recompense(etat)` reste abstraite.

Chaque labyrinthe concret contient sa configuration.

### Labyrinthe 3x3 (simple)

```text
0  1  2
3  4  5
6  7  8
```

```text
départ   : 0
arrivée  : 8
aucun mur, aucun piège
```

### Labyrinthe 5x5

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

### Labyrinthe 9x9 (murs et feu)

```text
S . . # . . . . .        0  1  2  3  4  5  6  7  8
. # . # . # # # .        9 10 11 12 13 14 15 16 17
. # . . . F . # .       18 19 20 21 22 23 24 25 26
. # # # # # . # .       27 28 29 30 31 32 33 34 35
. . . F . # . . .       36 37 38 39 40 41 42 43 44
# # # . # # # # .       45 46 47 48 49 50 51 52 53
. F . . . . . # .       54 55 56 57 58 59 60 61 62
. # # # # # F # .       63 64 65 66 67 68 69 70 71
. . . . F . . . G       72 73 74 75 76 77 78 79 80
```

```text
départ   : 0
arrivée  : 80
murs     : 3, 10, 12, 14, 15, 16, 19, 25, 28, 29, 30, 31, 32, 34, 41,
           45, 46, 47, 49, 50, 51, 52, 61, 64, 65, 66, 67, 68, 70
feux     : 23, 39, 55, 69, 76
```

Le chemin le plus court (16 déplacements) traverse le feu ; le plus court sans feu en fait 20. L'agent doit apprendre que 4 pas de détour (-0.4) coûtent moins qu'une case de feu (-1.0).

Récompenses (tous les labyrinthes) :

```text
case normale   : -0.1
piège / feu    : -1.0
arrivée        : +1.0
```

Les actions sont :

```text
0 = Haut
1 = Bas
2 = Gauche
3 = Droite
```

## Q-learning

`QLearning` reste une classe autonome et utilise un `EnvironnementDiscret`.

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
```

Le paramètre optionnel `seed` fixe le générateur aléatoire de l'agent : une même seed redonne exactement le même entraînement.

## Exécution

Depuis la racine du projet :

```bash
py -m experiences.entrainement
```

Génération des graphes (environ 2 minutes) :

```bash
py -m experiences.graphes
```

Pour chaque labyrinthe, 5 graphes sont écrits dans `resultats/<labyrinthe>/` :

| Fichier | Contenu |
|---|---|
| `1_convergence.png` | récompense cumulée, nombre de pas et epsilon par épisode |
| `2_epsilon_seeds.png` | impact d'epsilon sur la répétabilité : 4 stratégies (0.05, 0.2, 0.5 fixes, 1 -> 0.05 décroissant) x 10 seeds ; courbes moyenne ± écart-type et épisode de convergence de la politique gloutonne par seed |
| `3_surfaces_q.png` | Q(s, a) en 3D sur (ligne, colonne), une vue par action |
| `4_meilleure_action.png` | meilleure action argmax Q(s, a) dans chaque case et chemin glouton suivi |
| `5_carte_valeurs.png` | carte couleur de V(s) = max Q(s, a), normalisée entre 0 et 1 |
