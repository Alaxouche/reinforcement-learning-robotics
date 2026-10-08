# Projet RL — Q-learning sur environnements discrets

Le projet généralise les **environnements discrets** tout en gardant un seul algorithme : le Q-learning.

## Architecture

```text
reinforcement-learning-robotics/
├── src/
│   ├── environnements/
│   │   ├── environnement_discret.py
│   │   ├── labyrinthe.py
│   │   └── labyrinthe_4x4.py
│   │
│   └── algorithmes/
│       └── tabulaire/
│           └── q_learning.py
│
└── experiences/
    └── q_learning_tabulaire/
        └── entrainement.py
```

## Héritage des environnements

```text
EnvironnementDiscret
        ↑
    Labyrinthe
        ↑
  Labyrinthe4x4
```

`EnvironnementDiscret` définit le contrat commun : nombre d'états, nombre d'actions, `reset()`, `actions_possibles()` et `step()`.

`Labyrinthe` contient uniquement la mécanique commune aux labyrinthes : déplacements, limites de la grille, murs, calcul de l'état suivant, `reset()` et `step()`. La fonction `recompense(etat)` reste abstraite.

Les états sont numérotés de gauche à droite et de bas en haut :

```text
12  13  14  15
 8   9  10  11
 4   5   6   7
 0   1   2   3
```

Les actions sont :

```text
0 = Haut
1 = Bas
2 = Gauche
3 = Droite
```

## Labyrinthe 4 x 4

`Labyrinthe4x4` définit la configuration concrète utilisée pour les essais :

```text
départ   : 0
objectif : 15
murs     : 5, 14
feux     : 7, 9
```

Récompenses :

```text
déplacement normal : -0.1
feu                : -10
objectif           : +10
```

Les murs sont infranchissables. Une action impossible n'est jamais choisie par le Q-learning et la Q-value correspondante reste donc à sa valeur initiale.

Un chemin optimal attendu est :

```text
0 -> 1 -> 2 -> 6 -> 10 -> 11 -> 15
```

## Q-learning

`QLearning` reste une classe autonome et utilise n'importe quel `EnvironnementDiscret`.

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
```

## Exécution

```bash
py -m experiences.q_learning_tabulaire.entrainement
```
