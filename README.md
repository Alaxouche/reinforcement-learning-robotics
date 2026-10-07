# Projet RL — Q-learning tabulaire sur environnement discret

Le projet est volontairement généralisé **dans le cadre discret**.

L'objectif est de pouvoir ajouter plusieurs environnements discrets et plusieurs algorithmes tabulaires sans réécrire les parties communes.

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
│           ├── algorithme_tabulaire.py
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

C'est la classe abstraite commune à tous les environnements discrets.

Elle stocke :

- `nb_etats` ;
- `nb_actions`.

Elle impose aux classes filles :

- `reset()` ;
- `actions_possibles(etat)` ;
- `step(action)`.

Ainsi, le Q-learning n'a pas besoin de connaître les règles particulières du labyrinthe.

### `Labyrinthe`

`Labyrinthe` hérite de `EnvironnementDiscret`.

Il définit seulement ce qui est propre à une grille :

- la taille ;
- le départ et l'arrivée ;
- les déplacements ;
- les murs et feux ;
- l'état suivant ;
- les récompenses.

## Héritage des algorithmes

```text
AlgorithmeTabulaire
        ↑
        │
     QLearning
```

### `AlgorithmeTabulaire`

C'est la classe abstraite commune aux algorithmes tabulaires.

Elle stocke et gère :

- l'environnement discret ;
- le nombre d'épisodes ;
- le nombre maximal de pas ;
- l'historique des récompenses ;
- la table `Q` ;
- la recherche de la meilleure action ;
- la stratégie epsilon-greedy.

Elle impose :

- `choisir_action(etat)` ;
- `apprendre()`.

Un futur algorithme tabulaire comme SARSA pourra donc réutiliser cette classe.

### `QLearning`

`QLearning` hérite de `AlgorithmeTabulaire`.

Il contient uniquement les éléments propres au Q-learning :

- `alpha` ;
- `gamma` ;
- `epsilon` ;
- `epsilon_min` ;
- la décroissance d'epsilon ;
- la règle de mise à jour :

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
```

`QLearning` ne connaît pas la classe `Labyrinthe`. Il travaille avec n'importe quel objet qui hérite de `EnvironnementDiscret`.

## Expérience actuelle

Le fichier `experiences/q_learning_tabulaire/entrainement.py` assemble simplement un environnement et un algorithme :

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

Les hyperparamètres restent dans le fichier d'expérience.

## Installation

```bash
git clone https://github.com/Alaxouche/reinforcement-learning-robotics.git
cd reinforcement-learning-robotics
git switch hammou-qlearning
py -m pip install -r requirements.txt
```

## Exécution

```bash
py -m experiences.q_learning_tabulaire.entrainement
```

Le programme entraîne actuellement l'algorithme sans interface graphique. Les tests seront ajoutés séparément.
