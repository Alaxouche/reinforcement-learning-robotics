# Reinforcement Learning Robotics

Projet de Q-learning tabulaire sur deux environnements discrets :
un bras robotique 2D et des labyrinthes (5 x 5, 36 x 36).

## Structure simplifiee

```text
src/
  algorithmes/
    q_learning.py                  # Q-learning, exploration epsilon-greedy
  environnements/
    environnement_discret.py       # Interface des environnements
    bras_robotique/
      bras_robotique_2d.py         # Bras 2D, 4 ou 9 actions
      bras_robot_naive.py          # Recompense naive
      bras_robot_smart.py          # Recompense smart
      bras_robot_distance.py       # Recompense basee sur la distance
    labyrinthe/
      labyrinthe.py                # Mecanique de la grille
      labyrinthe_5x5.py
      labyrinthe_36x36.py
experiences/
  animation_bras.py                # Entrainement + animation du bras appris
requirements.txt
```

Les scripts d'etudes statistiques et les fonctions de generation de
graphiques ont ete retires pour garder uniquement les briques utiles
a l'algorithme, aux environnements et a l'animation.

## Installation

```bash
py -m pip install -r requirements.txt
```

## Animer le bras robotique

```bash
py -m experiences.animation_bras
```

Le menu permet de choisir 4 ou 9 actions, la fonction de recompense
(naive, smart, distance) et la pose de depart de l'animation.
Le programme entraine une Q-table, puis rejoue la politique apprise.

Execution sans menu :

```bash
py -m experiences.animation_bras --sans-menu --mode distance --actions 9 --episodes 5000
```

Enregistrer la Q-table ou la recharger :

```bash
py -m experiences.animation_bras --sans-menu --enregistrer-q experiences/q_tables/bras_distance_9.npy
py -m experiences.animation_bras --sans-menu --charger-q experiences/q_tables/bras_distance_9.npy
```

Exporter une animation GIF :

```bash
py -m experiences.animation_bras --sans-menu --gif --sans-fenetre
```

Les GIF sont places par defaut dans `experiences/animations/`.
Les Q-tables `.npy` et les GIF generes ne sont pas suivis par Git.

## Deux courbes pour vérifier l'apprentissage

Depuis la racine du dépôt :

```bash
py -m experiences.suivi_apprentissage
```

Par défaut : bras à 9 actions, récompense distance, 5 000 épisodes.
L'entraînement conserve **les départs aléatoires et l'epsilon dynamique**
habituels.

**Après chaque épisode d'entraînement**, la Q-table actuelle est évaluée
**sans epsilon** sur **tous les départs non terminaux**. On calcule :

1. **Taux de réussite global** : affiché dans le premier graphique tous les
   500 épisodes, avec 50 mouvements au maximum depuis chaque départ.
2. **Coût moyen global en mouvements**, affiché à chaque épisode dans le
   deuxième graphique. Un départ réussi coûte son nombre exact de mouvements.
   Un départ échoué coûte **51 points de pénalité** (limite = 50).
   Ainsi, aucune position difficile n'est écartée artificiellement du calcul.

L'indicateur est un **coût moyen pénalisé**, pas un nombre moyen de mouvements
avant réussite tant que le taux de réussite est inférieur à 100 %. Quand
toutes les positions réussissent, il devient la vraie moyenne des mouvements
pour atteindre la cible.

Les évaluations ne changent ni Q, ni epsilon, ni les tirages aléatoires.
Elles utilisent les transitions déterministes du bras et mettent en cache
les chemins pour éviter un temps de calcul prohibitif. Aucune recherche BFS,
aucun lissage, aucun dossier de résultats.

Avec une autre configuration :

```bash
py -m experiences.suivi_apprentissage --actions 4 --mode smart
```
