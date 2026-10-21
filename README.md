# Projet RL — Q-learning sur environnements discrets

Le projet généralise les environnements discrets tout en gardant un seul algorithme : le Q-learning.

## Architecture

```text
reinforcement-learning-robotics/
├── src/
│   ├── algorithmes/
│   │   └── q_learning.py
│   │
│   └── environnements/
│       ├── environnement_discret.py
│       └── labyrinthe/
│           ├── labyrinthe.py
│           └── labyrinthe_5x5.py
│
├── experiences/
│   ├── entrainement.py
│   └── graphiques.py
│
├── requirements.txt
└── README.md
```

Cette organisation reste volontairement simple :

- `src/algorithmes/` contient l'algorithme Q-learning ;
- `src/environnements/` contient les classes liées aux environnements ;
- `experiences/` contient le script qui assemble un environnement et l'algorithme pour lancer un entraînement.

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

`Labyrinthe5x5` contient la configuration concrète :

```text
 0   1   2   3   4
 5   6   7   8   9
10  11  12  13  14
15  16  17  18  19
20  21  22  23  24
```

```text
départ   : 0
arrivée  : 24
murs     : 3, 11, 13
pièges   : 7, 17
```

Récompenses :

```text
case normale : -0.1
piège        : -1.0
arrivée      : +1.0
```

Les actions sont :

```text
0 = Haut
1 = Bas
2 = Gauche
3 = Droite
```

## Q-learning

`QLearning` reste une classe autonome et utilise un `EnvironnementDiscret`.

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
```

## Exécution

Depuis la racine du projet :

```bash
py -m experiences.entrainement
```

## Graphiques d'apprentissage

`py -m experiences.entrainement` enregistre automatiquement 7 PNG dans
`experiences/resultats/` :

1. recompense cumulee par episode et moyenne glissante ;
2. nombre d'actions par episode ;
3. variation maximale de la Q-table ;
4. couverture cumulee des etats ;
5. epsilon et proportion observee d'exploration ;
6. moyenne de la valeur absolue de l'erreur TD ;
7. recompense moyenne lors de l'evaluation sans exploration.

Les mesures proviennent des attributs de `QLearning`, sans dependance
aux labyrinthes. Le dernier graphique repose sur des evaluations
periodiques de la politique gloutonne (5 episodes, aucune mise a jour
de la table Q). Le terme "couverture" designe le pourcentage d'etats
declares deja rencontres depuis le debut de l'entrainement.

Les graphiques produits sont ignores par Git et peuvent etre recrees
en relancant l'entrainement.
