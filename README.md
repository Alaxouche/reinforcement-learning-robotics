# Projet RL : apprentissage par renforcement appliqué à la robotique

Ce projet académique explore progressivement l'**apprentissage par renforcement** (Reinforcement Learning, RL), des premiers algorithmes tabulaires jusqu'à l'utilisation de réseaux de neurones et, à terme, à des applications en robotique.

Le **labyrinthe** est notre premier environnement d'expérimentation, pas la finalité du projet. Nous séparons donc les environnements des algorithmes pour pouvoir comparer différentes méthodes sur un même problème.

## Architecture actuelle

```text
reinforcement-learning-robotics/
├── src/
│   ├── environnements/
│   │   ├── environnement_discret.py    # Le contrat commun à tout environnement
│   │   └── labyrinthe.py               # Grille, obstacles, transitions, récompenses
│   ├── algorithmes/
│   │   └── tabulaire/
│   │       └── q_learning.py           # Q-learning avec table explicite
│   └── visualisation/
│       └── simulation_labyrinthe.py    # Animation Tkinter réutilisable
├── experiences/
│   └── q_learning_tabulaire/
│       ├── entrainement.py             # Lance l'apprentissage et affiche le résultat
│       └── simulation.py               # Lance la fenêtre animée
├── requirements.txt
├── README.md
└── .gitignore
```

Les répertoires Python utilisent les *namespace packages* de Python 3 : des fichiers `__init__.py` vides ne sont pas indispensables dans cette configuration. Lancer les commandes **depuis la racine du dépôt**.

## Les environnements

L'héritage va du plus général au plus précis :

```text
EnvironnementDiscret        le contrat : reset, actions_possibles, step
        │
    Labyrinthe              les règles de la grille
        │
Labyrinthe3x3 / 5x5 / 9x9   les trois configurations du projet
```

### `EnvironnementDiscret`

Classe abstraite (`ABC`) définissant ce que tout environnement doit fournir : les attributs `nb_etats` et `nb_actions`, et les trois méthodes `reset()`, `actions_possibles(etat)` et `step(action)`, marquées `@abstractmethod`.

Une classe fille à qui il manque une de ces méthodes ne peut pas être instanciée, et Python le signale immédiatement :

```text
TypeError: Can't instantiate abstract class EnvIncomplet with abstract method step
```

Python ne vérifie que les **noms** des méthodes, pas leurs paramètres ni leurs valeurs de retour. Les deux règles qu'il ne peut pas contrôler sont donc écrites dans les docstrings du fichier : `step` renvoie le triplet `(nouvel_etat, recompense, termine)`, et `actions_possibles` renvoie une liste **vide** sur un état terminal.

### `Labyrinthe`

Hérite de `EnvironnementDiscret` et porte toutes les règles du jeu. Son constructeur prend trois paramètres, `taille`, `murs` et `feux`, et appelle `super().__init__` pour enregistrer les dimensions du problème.

Les cases sont numérotées ligne par ligne, de 0 à `nb_etats - 1` :

```text
0 1 2
3 4 5
6 7 8
```

Le calcul des déplacements est écrit une seule fois, dans la méthode `voisine(etat, action)`, qui renvoie la case voisine ou `None` si le déplacement sort de la grille ou vise un mur. `actions_possibles`, `etat_suivant` et la vérification du labyrinthe reposent toutes sur elle.

À sa création, chaque grille est vérifiée : les obstacles sont dans la grille, ni le départ ni l'arrivée n'est un obstacle, et un parcours en largeur confirme qu'il existe au moins un chemin du départ à l'arrivée évitant les murs **et** les feux. Sinon, une `ValueError` est levée.

### Les trois configurations

| Identifiant | Taille | Obstacles | Chemin optimal |
|---|---:|---|---:|
| `3x3` | 3 × 3 | aucun | 4 pas |
| `5x5` | 5 × 5 | 6 murs | 8 pas |
| `9x9` | 9 × 9 | 13 murs, 4 feux | 16 pas |

```text
  3x3            5x5                  9x9

  D . .      D # . . .        D # # # . . . . .
  . . .      . # # . .        . . . # . F . . .
  . . A      . . . # .        . . . # . . . . .
             . . . # #        . . . # . F . . .
             . . . . A        . . . . . . . . .
                              . . . # . . . F .
                              . . . # . . . . .
D = départ    # = mur         . . . # # # # # .
A = arrivée   F = feu         . . F . . . . . A
```

Pour changer de labyrinthe, modifier la constante `CONFIGURATION_LABYRINTHE` en haut du script d'expérience voulu :

```python
CONFIGURATION_LABYRINTHE = "9x9"
```

Pour ajouter une configuration, écrire une classe fille puis l'inscrire dans le dictionnaire `LABYRINTHES`, en bas de `labyrinthe.py` :

```python
class Labyrinthe7x7(Labyrinthe):
    """Grille 7x7 avec des murs."""

    def __init__(self):
        super().__init__(taille=7, murs=[8, 9, 10], feux=[24])
```

## Première phase : Q-learning tabulaire

L'agent part de la case 0, cherche la dernière case et peut tenter de se déplacer en haut, en bas, à gauche ou à droite. Les actions qui sortent de la grille ou qui visent un mur sont exclues de la liste des actions possibles : l'agent ne peut pas les choisir.

| Événement | Récompense | Fin de l'épisode |
|---|---:|---|
| Arrivée | +1 | oui |
| Feu | −1 | oui |
| Autre déplacement | −0,1 | non |

L'algorithme apprend une **table** donnant une valeur pour chaque couple (état, action), à l'aide de la mise à jour de Bellman :

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
```

Un état terminal n'a pas de valeur future. La sélection des actions utilise une stratégie **epsilon-greedy** avec décroissance de l'exploration.

Les hyperparamètres sont indiqués explicitement dans chaque script d'expérience, et non fixés par défaut dans la classe `QLearning` : `alpha=0.1`, `gamma=0.9`, `epsilon=1.0`, `epsilon_min=0.05`, `decroissance=0.99`, `nb_episodes=1000` et `max_pas=30`. Ces valeurs suffisent aux trois labyrinthes.

## Installation

Prérequis : Python 3 et Tkinter (généralement inclus dans Python sur Windows).

```bash
git clone https://github.com/Alaxouche/reinforcement-learning-robotics.git
cd reinforcement-learning-robotics
python -m pip install -r requirements.txt
```

## Exécution

Entraîner l'agent, afficher sa table Q, son chemin et l'issue de l'épisode :

```bash
python -m experiences.q_learning_tabulaire.entrainement
```

```text
Chemin suivi : 0 -> 9 -> 18 -> 27 -> 28 -> 37 -> 38 -> 39 -> 40 -> 41 -> 42
               -> 51 -> 60 -> 61 -> 62 -> 71 -> 80
Pas : 16    Score : -0.50    Issue : arrivée
```

L'issue vaut `arrivée`, `feu` ou `trop de pas`.

Afficher l'animation Tkinter de l'agent entraîné :

```bash
python -m experiences.q_learning_tabulaire.simulation
```

L'entraînement a lieu avant l'ouverture de la fenêtre. Celle-ci propose les boutons **Démarrer** et **Recommencer**, et indique la fin de l'épisode : `arrivée !`, `feu : épisode perdu` ou `trop de pas`. Sous Windows, la fenêtre s'ouvre parfois derrière l'éditeur. Pour quitter, la fermer avec la croix, pas avec Ctrl+C.
