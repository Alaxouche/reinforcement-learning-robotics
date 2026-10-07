# Projet RL — Apprentissage par renforcement appliqué à la robotique

Ce projet académique est construit pour pouvoir faire évoluer séparément les **environnements** et les **algorithmes d'apprentissage par renforcement**.

Le labyrinthe 3 × 3 et le Q-learning tabulaire sont seulement les premières implémentations concrètes. L'architecture est volontairement générale afin de pouvoir ajouter ensuite d'autres environnements, d'autres algorithmes tabulaires, des méthodes profondes et des applications robotiques.

## Architecture

```text
reinforcement-learning-robotics/
├── src/
│   ├── environnements/
│   │   ├── environnement.py
│   │   ├── environnement_discret.py
│   │   └── labyrinthe.py
│   │
│   └── algorithmes/
│       ├── algorithme.py
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

## Hiérarchie des environnements

```text
Environnement
      ↑
      │
EnvironnementDiscret
      ↑
      │
  Labyrinthe
```

### `Environnement`

`Environnement` est la classe abstraite la plus générale.

Elle impose seulement :

- `reset()` : recommencer un épisode ;
- `step(action)` : exécuter une action et renvoyer le nouvel état, la récompense et l'indicateur de fin.

Elle ne suppose **ni grille, ni nombre fini d'états, ni nombre fini d'actions**.

Un futur environnement continu ou robotique pourra donc hériter directement de cette classe.

### `EnvironnementDiscret`

`EnvironnementDiscret` hérite de `Environnement`.

Il ajoute les informations nécessaires aux méthodes tabulaires :

- `nb_etats` ;
- `nb_actions` ;
- `actions_possibles(etat)`.

### `Labyrinthe`

`Labyrinthe` hérite de `EnvironnementDiscret`.

Il ne contient que les règles propres à la grille :

- taille ;
- départ et arrivée ;
- déplacements haut, bas, gauche, droite ;
- murs et feux ;
- calcul de l'état suivant ;
- récompenses.

## Hiérarchie des algorithmes

```text
AlgorithmeRL
      ↑
      │
AlgorithmeTabulaire
      ↑
      │
   QLearning
```

### `AlgorithmeRL`

`AlgorithmeRL` est la classe abstraite générale des algorithmes d'apprentissage par renforcement.

Elle stocke ce qui est commun à une expérience :

- l'environnement ;
- le nombre d'épisodes ;
- le nombre maximal de pas par épisode ;
- l'historique des récompenses.

Elle impose :

- `choisir_action(etat)` ;
- `apprendre()`.

Elle ne suppose pas l'existence d'une table Q.

### `AlgorithmeTabulaire`

`AlgorithmeTabulaire` hérite de `AlgorithmeRL`.

Il exige un `EnvironnementDiscret` et fournit les éléments communs aux méthodes tabulaires :

- création de la table `Q` ;
- `meilleure_action(etat)` ;
- stratégie générique `action_epsilon_greedy(etat, epsilon)`.

Cela permettra par exemple d'ajouter plus tard un algorithme **SARSA** sans recopier toute cette partie.

### `QLearning`

`QLearning` hérite de `AlgorithmeTabulaire`.

Il contient uniquement ce qui est propre au Q-learning :

- `alpha` ;
- `gamma` ;
- `epsilon` ;
- `epsilon_min` ;
- la décroissance d'epsilon ;
- la règle de mise à jour de Bellman.

La mise à jour utilisée est :

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
```

Ainsi, **QLearning ne connaît jamais la classe Labyrinthe**. Il travaille uniquement avec l'interface d'un environnement discret.

## Expérience actuelle

Le fichier :

```text
experiences/q_learning_tabulaire/entrainement.py
```

est volontairement concret : c'est lui qui assemble un environnement et un algorithme.

Actuellement :

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

Les hyperparamètres restent donc dans l'expérience et ne sont pas imposés par les classes abstraites.

## Pourquoi cette architecture ?

La séparation permet de combiner indépendamment les deux côtés.

Par exemple, à terme :

```text
Environnements
├── Labyrinthe
├── EnvironnementRobot
└── autre environnement

Algorithmes
├── QLearning
├── SARSA
└── DQN
```

On pourra ajouter une nouvelle classe sans réécrire les autres parties du projet, tant qu'elle respecte l'interface attendue.

Le Q-learning tabulaire impose naturellement un environnement discret, tandis qu'un futur algorithme profond pourra hériter directement de `AlgorithmeRL` et fonctionner avec un environnement plus général.

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

Le programme entraîne actuellement l'algorithme sans interface graphique et sans affichage console automatique. Les tests et les outils d'évaluation seront ajoutés séparément.

> Les modifications de cette version sont développées sur la branche `hammou-qlearning`.
