# Projet RL — Apprentissage par renforcement appliqué à la robotique

Ce projet académique explore progressivement l'**apprentissage par renforcement** (Reinforcement Learning, RL), des premiers algorithmes tabulaires jusqu'à l'utilisation de réseaux de neurones et, à terme, à des applications en robotique.

Le **labyrinthe** est notre premier environnement d'expérimentation, pas la finalité du projet. Nous séparons donc les environnements des algorithmes pour pouvoir comparer différentes méthodes sur un même problème.

## Architecture actuelle

```text
qlearning-labyrinthe-3x3/
├── src/
│   ├── environnements/
│   │   └── labyrinthe.py                # Grille, actions, transitions, récompenses
│   ├── algorithmes/
│   │   └── tabulaire/
│   │       └── q_learning.py            # Q-learning avec table explicite
│   └── visualisation/
│       └── simulation_labyrinthe.py     # Animation Tkinter réutilisable
├── experiences/
│   └── q_learning_tabulaire/
│       ├── entrainement.py              # Lance l'apprentissage 3 × 3
│       └── simulation.py                # Lance la simulation du même exemple
├── requirements.txt
├── README.md
└── .gitignore
```

Les répertoires Python utilisent les *namespace packages* de Python 3 : des fichiers `__init__.py` vides ne sont pas indispensables dans cette configuration. Lancer les commandes **depuis la racine du dépôt**.

## Première phase : Q-learning tabulaire

L'environnement initial est un labyrinthe carré **3 × 3, sans mur ni feu**. L'agent part de la case 0, cherche la dernière case et peut tenter de se déplacer en haut, en bas, à gauche ou à droite. Les actions qui sortent de la grille sont exclues de la liste des actions possibles : l'agent ne peut pas les choisir.

| Événement | Récompense |
|---|---:|
| Arrivée | +1 |
| Autre déplacement | −0,1 |

L'algorithme apprend une **table** donnant une valeur pour chaque couple (état, action), à l'aide de la mise à jour de Bellman :

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
```

L'état final n'a pas de valeur future. La sélection des actions utilise une stratégie **epsilon-greedy** avec décroissance de l'exploration.

Les hyperparamètres sont indiqués explicitement dans chaque script d'expérience (et non fixés par défaut dans la classe `QLearning`). Dans cet exemple : `alpha=0.1`, `gamma=0.9`, `epsilon=1.0`, `epsilon_min=0.05`, `decroissance=0.99`, `nb_episodes=1000` et `max_pas=30`. L'environnement fournit `nb_etats`, `nb_actions`, `reset()`, `actions_possibles(etat)` et `step(action)`.

## Installation

Prérequis : Python 3 et Tkinter (généralement inclus dans Python sur Windows).

```bash
git clone https://github.com/Alaxouche/qlearning-labyrinthe-3x3.git
cd qlearning-labyrinthe-3x3
python -m pip install -r requirements.txt
```

## Exécution

Entraîner l'agent, afficher sa table Q et son chemin :

```bash
python -m experiences.q_learning_tabulaire.entrainement
```

Afficher l'animation Tkinter de l'agent entraîné :

```bash
python -m experiences.q_learning_tabulaire.simulation
```

La fenêtre propose les boutons **Démarrer** et **Recommencer**.

### Choisir un labyrinthe

Les trois configurations disponibles sont construites par `creer_labyrinthe` dans
`src/environnements/labyrinthe.py` :

| Identifiant | Taille | Obstacles |
|---|---:|---|
| `3x3` | 3 × 3 | aucun |
| `5x5` | 5 × 5 | murs |
| `9x9` | 9 × 9 | murs et feux |

Pour changer de labyrinthe, modifier la constante `CONFIGURATION_LABYRINTHE`
dans le script d'expérience voulu, par exemple :

```python
CONFIGURATION_LABYRINTHE = "9x9"
```

Les feux donnent une récompense de `-1` et terminent l'épisode. Les murs ne sont
pas traversables. Chaque configuration est vérifiée à sa création : il existe
toujours au moins un chemin du départ à l'arrivée qui évite les murs et les feux.

## Progression envisagée

| Phase | Sujet | Statut |
|---|---|---|
| 1 | Q-learning **tabulaire** sur labyrinthe simple | Première version disponible |
| 2 | Environnements avec murs, feux, obstacles et mesures de performance | À développer |
| 3 | **Deep Q-Network (DQN)** : un réseau de neurones approxime les valeurs `Q(s,a; θ)` à la place d'une table explicite | À développer |
| 4 | Navigation dans des environnements plus complexes et robotique simulée | À développer |

À mesure que le projet progressera, les nouveaux algorithmes seront ajoutés dans `src/algorithmes/` (par exemple `profond/dqn.py`) et leurs protocoles dans `experiences/`. **Aucun fichier DQN fictif n'est créé à ce stade.**

> Le titre du projet a été élargi ; le nom et l'URL du dépôt GitHub restent inchangés tant que son propriétaire ne les modifie pas.
