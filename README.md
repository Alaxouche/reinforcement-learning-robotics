# Projet RL — Q-learning sur environnements discrets

Le projet généralise les **environnements discrets** tout en gardant un seul algorithme : le Q-learning.

## Architecture

```text
reinforcement-learning-robotics/
├── src/
│   ├── environnements/
│   │   ├── environnement_discret.py
│   │   ├── labyrinthe.py
│   │   └── labyrinthe_5x5.py
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
  Labyrinthe5x5
```

`EnvironnementDiscret` définit le contrat commun : nombre d'états, nombre d'actions, `reset()`, `actions_possibles()` et `step()`.

`Labyrinthe` contient la mécanique commune aux labyrinthes : déplacements, limites de la grille, murs, calcul de l'état suivant, `reset()` et `step()`. La fonction `recompense(etat)` reste abstraite.

Les états du labyrinthe actuel sont numérotés de gauche à droite et de haut en bas :

```text
 0   1   2   3   4
 5   6   7   8   9
10  11  12  13  14
15  16  17  18  19
20  21  22  23  24
```

Les actions sont :

```text
0 = Haut
1 = Bas
2 = Gauche
3 = Droite
```

## Labyrinthe 5 x 5

`Labyrinthe5x5` définit la configuration concrète utilisée pour les essais :

```text
départ   : 0
arrivée  : 24
murs     : 3, 11, 13
pièges   : 7, 17
```

Récompenses actuellement utilisées :

```text
case normale : -0.1
piège        : -1.0
arrivée      : +1.0
```

Les murs sont infranchissables. Les pièges restent accessibles mais donnent une récompense négative.

## Q-learning

`QLearning` reste une classe autonome et utilise n'importe quel `EnvironnementDiscret`.

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
```

## Exécution

```bash
py -m experiences.q_learning_tabulaire.entrainement
```
