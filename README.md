# Projet RL — Q-learning sur environnements discrets

Le projet généralise les **environnements discrets** tout en gardant un seul algorithme : le Q-learning.

## Architecture

```text
reinforcement-learning-robotics/
├── src/
│   ├── environnements/
│   │   ├── environnement_discret.py
│   │   ├── labyrinthe.py
│   │   └── labyrinthe_simple.py
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
 LabyrintheSimple
```

`EnvironnementDiscret` définit le contrat commun : nombre d'états, nombre d'actions, `reset()`, `actions_possibles()` et `step()`.

`Labyrinthe` est une classe de base pour les différents labyrinthes. Elle contient la mécanique commune : déplacements sur la grille, vérification des limites et des murs, calcul de l'état suivant, `reset()` et `step()`.

La méthode `recompense(etat)` est abstraite dans `Labyrinthe`. Chaque labyrinthe concret doit donc définir sa propre fonction de récompense.

`LabyrintheSimple` choisit actuellement une grille sans mur, un départ en case 0, une arrivée sur la dernière case et la récompense suivante :

```text
arrivée        -> +1.0
autre état     -> -0.1
```

Ainsi, un futur labyrinthe pourra avoir d'autres murs et une autre fonction de récompense sans modifier `Labyrinthe` ni `QLearning`.

## Q-learning

`QLearning` reste une classe autonome. Elle utilise n'importe quel `EnvironnementDiscret` et contient directement la table Q, epsilon-greedy et la mise à jour de Bellman.

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
```

## Exécution

```bash
py -m experiences.q_learning_tabulaire.entrainement
```
