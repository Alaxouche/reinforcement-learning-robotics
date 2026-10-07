# Projet RL — Apprentissage par renforcement appliqué à la robotique

Ce projet académique explore progressivement l'**apprentissage par renforcement** (Reinforcement Learning, RL), des premiers algorithmes tabulaires jusqu'à l'utilisation de réseaux de neurones et, à terme, à des applications en robotique.

Le **labyrinthe** est notre premier environnement d'expérimentation, pas la finalité du projet. Les environnements suivent maintenant une hiérarchie par héritage : une classe mère abstraite `Environnement` définit l'interface commune, puis les environnements concrets comme `Labyrinthe` en héritent.

## Architecture actuelle

```text
reinforcement-learning-robotics/
├── src/
│   ├── environnements/
│   │   ├── environnement.py             # Classe mère abstraite
│   │   └── labyrinthe.py                # Classe fille : grille, transitions, récompenses
│   ├── algorithmes/
│   │   └── tabulaire/
│   │       └── q_learning.py            # Q-learning avec table explicite
├── experiences/
│   └── q_learning_tabulaire/
│       └── entrainement.py              # Création de l'agent et lancement de l'apprentissage
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

Les hyperparamètres sont indiqués explicitement dans l'unique script d'expérience (et non fixés par défaut dans la classe `QLearning`). Dans cet exemple : `alpha=0.1`, `gamma=0.9`, `epsilon=1.0`, `epsilon_min=0.05`, `decroissance=0.99`, `nb_episodes=1000` et `max_pas=30`.

## Héritage des environnements

La classe abstraite `Environnement` définit le contrat minimal attendu par `QLearning` : `nb_etats`, `nb_actions`, `reset()`, `actions_possibles(etat)` et `step(action)`. Elle ne décrit aucun problème concret.

`Labyrinthe` hérite de cette classe avec `class Labyrinthe(Environnement)` et appelle `super().__init__(...)` pour initialiser les attributs communs `nb_etats` et `nb_actions`. Les règles propres au labyrinthe restent dans la classe fille : déplacements, arrivée, récompenses et obstacles.

Cette organisation permet d'ajouter plus tard d'autres environnements (`EnvironnementRobot`, autres grilles, etc.) sans modifier l'algorithme `QLearning`.

## Déroulement d'un épisode d'apprentissage

Le fichier `experiences/q_learning_tabulaire/entrainement.py` crée un environnement concret (`Labyrinthe`), le transmet à `QLearning`, puis lance `agent.apprendre()`. Un épisode correspond à une partie complète.

1. **Réinitialisation — `etat = self.env.reset()`** : l'environnement revient à son état initial.
2. **Choix d'une action — `action = self.choisir_action(etat)`** : stratégie epsilon-greedy parmi les actions autorisées.
3. **Transition — `self.env.step(action)`** : l'environnement renvoie le nouvel état, la récompense et l'indicateur de fin.
4. **Mise à jour de Bellman** : une seule valeur `Q[etat, action]` est corrigée.
5. **Suite ou fin** : l'agent continue jusqu'à la fin de l'épisode ou jusqu'à `max_pas`.

Après chaque épisode, le score est conservé et `epsilon` diminue progressivement.

## Installation

Prérequis : Python 3.

```bash
git clone https://github.com/Alaxouche/reinforcement-learning-robotics.git
cd reinforcement-learning-robotics
python -m pip install -r requirements.txt
```

## Exécution

Pour lancer l'entraînement :

```bash
py -m experiences.q_learning_tabulaire.entrainement
```

## Progression envisagée

| Phase | Sujet | Statut |
|---|---|---|
| 1 | Q-learning **tabulaire** sur labyrinthe simple | Première version disponible |
| 2 | Environnements avec murs, feux, obstacles et mesures de performance | À développer |
| 3 | **Deep Q-Network (DQN)** : un réseau de neurones approxime les valeurs `Q(s,a; θ)` à la place d'une table explicite | À développer |
| 4 | Navigation dans des environnements plus complexes et robotique simulée | À développer |

À mesure que le projet progressera, les nouveaux algorithmes seront ajoutés dans `src/algorithmes/` (par exemple `profond/dqn.py`) et leurs protocoles dans `experiences/`. **Aucun fichier DQN fictif n'est créé à ce stade.**

> Les modifications pédagogiques de cette version sont effectuées sur la branche `hammou-qlearning`, indépendamment de `main`.
