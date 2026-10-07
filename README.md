# Projet RL — Q-learning sur environnement discret

Le projet généralise uniquement les **environnements discrets**.

L'algorithme reste volontairement simple : il n'existe qu'une seule classe `QLearning`.

## Architecture

```text
reinforcement-learning-robotics/
├── src/
│   ├── environnements/
│   │   ├── environnement_discret.py
│   │   └── labyrinthe.py
│   │
│   └── algorithmes/
│       └── tabulaire/
│           └── q_learning.py
│
├── experiences/
│   └── q_learning_tabulaire/
│       └── entrainement.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Héritage des environnements

```text
EnvironnementDiscret
        ↑
        │
    Labyrinthe
```

### `EnvironnementDiscret`

C'est la classe abstraite commune aux environnements discrets.

Elle contient :

- `nb_etats` ;
- `nb_actions`.

Elle impose aux classes filles :

- `reset()` ;
- `actions_possibles(etat)` ;
- `step(action)`.

### `Labyrinthe`

`Labyrinthe` hérite de `EnvironnementDiscret` et définit seulement les règles propres à la grille :

- taille ;
- départ et arrivée ;
- déplacements ;
- murs et feux ;
- état suivant ;
- récompenses.

## Q-learning

Il n'y a pas de classe mère pour les algorithmes.

La classe `QLearning` contient directement :

- l'environnement ;
- les hyperparamètres ;
- la table `Q` ;
- `meilleure_action()` ;
- `choisir_action()` avec epsilon-greedy ;
- `apprendre()` avec la mise à jour de Bellman.

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
```

`QLearning` ne dépend pas de `Labyrinthe` : il accepte n'importe quel objet qui hérite de `EnvironnementDiscret`.

## Expérience actuelle

```python
environnement = Labyrinthe(taille=3)

algorithme = QLearning(
    environnement=environnement,
    alpha=0.1,
    gamma=0.9,
    epsilon=1.0,
    epsilon_min=0.05,
    decroissance=0.99,
    nb_episodes=1000,
    max_pas=30
)

algorithme.apprendre()
```

## Exécution

```bash
py -m experiences.q_learning_tabulaire.entrainement
```
