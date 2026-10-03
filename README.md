# Labyrinth 3x3 Q-Learning

## Les fichiers
 
- `labyrinthe.py` : la classe `Labyrinthe` (l'environnement). C'est le labyrinthe simple sans mur ni feu, et le modèle à suivre pour les autres.
- `apprentissage.py` : la classe `QLearning` (l'apprentissage). Elle marche avec n'importe quel labyrinthe. Lancé directement, il affiche le tableau Q et le chemin trouvé.
- `simulation.py` : une petite fenêtre Tkinter où on voit l'agent entraîné suivre son chemin, case par case.

Les paramètres se donnent à la création de l'objet `QLearning`, par exemple `QLearning(lab, nb_episodes=2000, max_pas=200)`. Ceux qu'on ne donne pas gardent leur valeur par défaut :

| Paramètre | Valeur | Rôle |
|---|---|---|
| `alpha` | 0.1 | vitesse d'apprentissage |
| `gamma` | 0.9 | importance des récompenses futures |
| `epsilon_debut` | 1.0 | exploration au début |
| `epsilon_min` | 0.05 | exploration minimale |
| `decroissance` | 0.99 | vitesse de baisse d'epsilon |
| `nb_episodes` | 500 | nombre de parties d'entraînement |
| `max_pas` | 50 | nombre de pas maximum par partie |
| `graine` | 0 | pour avoir les mêmes résultats à chaque lancement |

Un grand labyrinthe demande plus d'épisodes et plus de pas par partie.

## Choisir un labyrinthe

`labyrinthe.py` propose trois labyrinthes prédéfinis :

| Nom | Contenu |
|---|---|
| `simple_3x3` | grille 3 × 3 vide |
| `murs_5x5` | grille 5 × 5 avec 6 murs |
| `murs_feux_9x9` | grille 9 × 9 avec 27 murs et 5 feux |

Les murs et les feux sont placés au hasard à chaque lancement, en gardant toujours un chemin sans feu jusqu'à la sortie. Un mur bloque l'agent comme un bord ; un feu donne une récompense de -1. Pour retrouver toujours le même labyrinthe, on donne une graine : `creer_labyrinthe("murs_5x5", graine=42)`.

Pour changer de labyrinthe, il suffit de changer les deux lignes `lab = ...` et `qlearning = ...` en bas de `apprentissage.py` et en haut de `simulation.py`.

## Ajouter un labyrinthe

On crée une classe fille de `Labyrinthe` (comme `LabyrintheAleatoire`), puis on l'ajoute au dictionnaire `LABYRINTHES`. La classe `QLearning` n'utilise que :

- les attributs `taille`, `nb_etats`, `depart`, `arrivee`, `actions`, `noms_actions` ;
- les méthodes `etat_suivant(etat, action)` (la case où l'on arrive) et `recompense(etat)` (la récompense en arrivant sur cette case).

La simulation utilise en plus les listes `murs` et `feux` (numéros de cases) pour les colorier.

## Lancer le projet
 
Il faut Python 3 avec `numpy` (Tkinter est fourni avec Python).
 
```
pip install numpy
```
 
Pour l'entraînement :
 
```
python apprentissage.py
```
 
Pour la simulation (elle affiche aussi le tableau Q) :
 
```
python simulation.py
```
 
Cliquez sur **Démarrer** pour lancer l'agent et sur **Recommencer** pour le remettre au départ. Pour quitter, fermez simplement la fenêtre. Sous Windows, elle s'ouvre parfois derrière l'éditeur.

## Fonctionnement
 
L'agent garde un tableau `Q` qui donne, pour chaque case et chaque direction (haut, bas, gauche, droite), une note : « est-ce que c'est une bonne idée d'aller par là ? ».
 
À chaque déplacement, il met cette note à jour avec la formule du Q-learning :
 
```
Q(s, a) <- Q(s, a) + alpha * (r + gamma * max Q(s', .) - Q(s, a))
```
 
Les récompenses :
- +1 quand il atteint la sortie ;
- -0.1 pour chaque pas, pour qu'il cherche le chemin le plus court.
Pour ne pas rester bloqué sur ses premières idées, l'agent explore parfois au hasard. C'est le paramètre `epsilon` : il vaut 1 au début (tout au hasard), puis il diminue à chaque épisode jusqu'à 0.05. Au début l'agent découvre, à la fin il exploite ce qu'il a appris.

## Résultat
 
Après 500 épisodes, l'agent trouve un chemin de 4 pas, le plus court possible, par exemple `0 -> 3 -> 4 -> 5 -> 8`. Dans la simulation, le score final est de 0.7 (trois pas à -0.1, puis +1 à l'arrivée).
