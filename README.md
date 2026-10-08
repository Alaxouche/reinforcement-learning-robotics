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

`Labyrinthe` contient la mécanique commune aux labyrinthes : déplacements dans une grille, vérification des limites, prise en compte des murs, calcul de l'état suivant et exécution d'une action.

Il ne décide pas qu'un labyrinthe particulier possède zéro mur : la liste des murs lui est fournie lors de sa construction.

`LabyrintheSimple` est l'environnement concret actuellement utilisé. Il choisit une grille sans mur, un départ en case 0 et une arrivée sur la dernière case.

La notion de `feux` a été retirée tant qu'aucun comportement associé n'est implémenté.

## Q-learning

`QLearning` reste une classe autonome. Elle utilise n'importe quel `EnvironnementDiscret` et contient directement la table Q, epsilon-greedy et la mise à jour de Bellman.

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
```

## Exécution

```bash
py -m experiences.q_learning_tabulaire.entrainement
```
