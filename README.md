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

`py -m experiences.entrainement` enregistre automatiquement 6 PNG dans
`experiences/resultats/` :

1. recompense cumulee par episode et moyenne glissante ;
2. nombre d'actions par episode ;
3. variation maximale de la Q-table ;
4. couverture cumulee des etats ;
5. moyenne de la valeur absolue de l'erreur TD ;
6. recompense moyenne lors de l'evaluation sans exploration.

Les mesures proviennent des attributs de `QLearning`, sans dependance
aux labyrinthes. Le dernier graphique repose sur des evaluations
periodiques de la politique gloutonne (5 episodes, aucune mise a jour
de la table Q). Le terme "couverture" designe le pourcentage d'etats
declares deja rencontres depuis le debut de l'entrainement.

Les graphiques produits sont ignores par Git et peuvent etre recrees
en relancant l'entrainement.

## Deuxieme environnement : bras robotique 2D

Le robot comporte deux segments de longueurs L1 et L2 fixes.
Theta1 est mesure par rapport a l'horizontale, theta2 est l'angle
du second segment par rapport au premier. Les coordonnees de
l'effecteur sont :

```text
x = L1*cos(theta1) + L2*cos(theta1+theta2)
y = L1*sin(theta1) + L2*sin(theta1+theta2)
```

Les angles sont discrets. Les quatre actions consistent a augmenter
ou diminuer l'un des deux angles d'un pas (10 degres par defaut).
On impose des butees articulaires et on interdit les poses dont le
coude ou l'effecteur passe sous le sol y = 0.

La cible M0=(x0,y0) est **fixe pendant chaque entrainement**.
Le bras part de (theta1,theta2)=(90,0) degres et tente d'atteindre
M0=(1,1) avec une tolerance de 0.15 unite de longueur.
Le simulateur verifie qu'une position admissible peut atteindre
la cible sur la grille d'angles choisie.

La recompense pour une action est :
`2*(distance_avant-distance_apres)-0.02`, plus `10` en cas
de reussite. Il s'agit d'un modele cinematique simplifie :
il n'inclut pas la dynamique des moteurs ni les collisions pendant
la rotation continue entre deux poses.

### Lancer le bras robotique

```bash
py -m experiences.entrainement_bras_robotique
```

Les six graphiques d'apprentissage generiques et un septieme graphique
representant le bras et la trajectoire de son effecteur seront dans
`experiences/resultats/bras_robotique/`, separes des resultats du labyrinthe.

Les parametres (longueurs, angles, position cible, tolerance et nombre
d'episodes) sont modifiables dans
`experiences/entrainement_bras_robotique.py`.

Pour executer les tests de la cinematique et des contraintes :

```bash
py -m unittest discover -s tests
```

A noter : si la cible change entre les episodes d'un meme entrainement,
il faut integrer sa position dans l'etat RL ; cette premiere version
apprend uniquement pour une cible fixe.
