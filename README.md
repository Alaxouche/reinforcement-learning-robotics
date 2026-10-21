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
│       ├── labyrinthe/
│       │   ├── labyrinthe.py
│       │   └── labyrinthe_5x5.py
│       └── bras_robotique/
│           └── bras_robotique_2d.py
│
├── experiences/
│   ├── entrainement.py
│   ├── entrainement_bras_robotique.py
│   ├── graphiques.py
│   └── resultats/
│       ├── labyrinthes/
│       └── bras_robotique/
│
├── tests/
│   └── test_bras_robotique.py
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
`experiences/resultats/labyrinthes/` :

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
en relancant l'entrainement. Les labyrinthes enregistrent les images dans
`experiences/resultats/labyrinthes/` et le bras robotique dans
`experiences/resultats/bras_robotique/`. Au prochain entrainement
du labyrinthe, les anciens PNG generes directement dans
`experiences/resultats/` sont deplaces dans le dossier `labyrinthes/`.

## Deuxieme environnement : bras robotique 2D

Le robot comporte deux segments de longueurs L1 et L2 fixes.
Theta1 est mesure par rapport a l'horizontale, theta2 est l'angle
du second segment par rapport au premier. Les coordonnees de
l'effecteur sont :

```text
x = L1*cos(theta1) + L2*cos(theta1+theta2)
y = L1*sin(theta1) + L2*sin(theta1+theta2)
```

Les angles sont discrets et independamment libres de tourner sur 360 degres.
Les quatre actions augmentent ou diminuent l'un des angles de 10 degres
(par defaut), avec bouclage modulo 360 : 350 + 10 = 0 degre.
Les angles vont de 0 a 350 degres inclus (360 equivaut a 0).
Cela produit 36 x 36 = 1296 etats, sans contraintes de sol ni de quadrant.
La cible peut avoir des coordonnees x et y negatives.

La cible M0=(x0,y0) est **fixe pendant chaque entrainement**.
Le bras part de (theta1,theta2)=(90,0) degres et tente d'atteindre
M0=(1,1) avec une tolerance de 0.15 unite de longueur.
Le simulateur verifie qu'une configuration discretisee peut atteindre
la cible dans la tolerance choisie. La cible peut etre situee dans
n'importe lequel des quatre quadrants, mais la portee physique
reste limitee par les longueurs L1 et L2 : la distance a l'origine
est comprise entre abs(L1-L2) et L1+L2 (avant discretisation).

Deux recompenses sont disponibles pour le meme simulateur :

- **Naive (parcimonieuse)** : `+10` lorsque la cible est atteinte,
  `-0.02` sinon. Elle indique uniquement la reussite et le cout
  des actions.
- **Smart (guidee par la distance)** : `+10` lorsque la cible est
  atteinte ; sinon `distance_avant - distance_apres - 0.02`.
  Elle favorise le rapprochement sans coefficient multiplicateur.
  Exemple : de 1.0 a 0.8 unite, la recompense vaut `0.18`.
  Le terme `-0.02` penalise les mouvements inutiles.

Les deux variantes utilisent les memes etats, actions, angles, cibles
et hyperparametres. Seule la fonction de recompense change.
La recompense smart est plus informative pendant l'entrainement,
mais n'assure pas necessairement le chemin le plus court.
Les courbes de recompenses des deux modes ne sont pas directement
comparables numeriquement car leurs definitions sont differentes.

Il s'agit d'un modele cinematique simplifie :
il n'inclut pas la dynamique des moteurs ni les collisions pendant
la rotation continue entre deux poses. Aucune butee articulaire
n'est imposee dans cette simulation.

### Lancer les robots (naif et smart)

Les deux entrainements sont executes automatiquement, l'un apres l'autre :

```bash
py -m experiences.entrainement_bras_robotique
```

On peut aussi choisir un seul mode :

```bash
py -m experiences.entrainement_bras_robotique --mode naive
py -m experiences.entrainement_bras_robotique --mode smart
```

Pour chaque mode, six graphiques d'apprentissage et une image du
bras/trajectoire sont enregistres independamment :

```text
experiences/resultats/bras_robotique/
  naive/
    01_recompenses.png ... 07_trajectoire_bras.png
  smart/
    01_recompenses.png ... 07_trajectoire_bras.png
```

Pour comparer les agents, prioriser le succes de l'evaluation sans
exploration, le nombre de mouvements et la distance finale a la cible,
plutot que la recompense brute : elle n'a pas la meme definition.

Les parametres (longueurs, pas angulaire, position cible, tolerance et
nombre d'episodes) sont modifiables dans
`experiences/entrainement_bras_robotique.py`.

Pour executer les tests de la cinematique et des contraintes :

```bash
py -m unittest discover -s tests
```

A noter : si la cible change entre les episodes d'un meme entrainement,
il faut integrer sa position dans l'etat RL ; cette premiere version
apprend uniquement pour une cible fixe.
