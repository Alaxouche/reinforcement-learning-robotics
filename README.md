# Reinforcement Learning Robotics

Projet de Q-learning tabulaire et de DQN PyTorch sur deux environnements :
un bras robotique 2D et des labyrinthes (5 x 5, 36 x 36).

## Structure simplifiee

```text
src/
  algorithmes/
    q_learning.py                  # Q-learning tabulaire
    dqn.py                         # Réseau de neurones (PyTorch), replay et cible
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
  animation_bras.py                # Animation du bras, Q-table de référence
  suivi_apprentissage.py           # Courbes pour Q-learning tabulaire
  dqn_bras.py                      # Entrainement, courbes et animation du DQN
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


## Nouveau : DQN avec réseau de neurones PyTorch

Le Q-learning tabulaire reste disponible pour servir de **référence**. Le DQN
reprend exactement le même bras (4 ou 9 actions), les départs aléatoires,
les fonctions de récompense `naive`, `smart`, `distance`, ainsi que
l'exploration epsilon-greedy. Il n'utilise pas de Q-table.

**Entrées du réseau** : cos(theta1), sin(theta1), cos(theta2), sin(theta2).
**Architecture** : 4 -> 64 -> 64 -> 9 valeurs Q (ou 4 selon les actions).
Le réseau apprend par descente de gradient sur les expériences mémorisées.
Un réseau cible est utilisé pour rendre l'entraînement plus stable.

Installer PyTorch et les dépendances :

```powershell
py -m pip install -r requirements.txt
```

Entraîner 5 000 épisodes et afficher les deux graphiques :

```powershell
py -m experiences.dqn_bras --episodes 5000
```

Les tests, sans epsilon, portent toujours sur tous les départs : taux de
réussite (%) et coût moyen pénalisé (51 pour chaque échec). Ils sont
effectués par défaut après chaque épisode, comme dans l'étude tabulaire.
Pour accélérer l'entraînement en testant tous les 100 épisodes :

```powershell
py -m experiences.dqn_bras --episodes 5000 --eval-tous 100
```

Sauvegarder les poids et animer le bras avec le réseau appris :

```powershell
py -m experiences.dqn_bras --episodes 5000 --sauvegarder experiences/modeles_dqn/bras_distance_9.pt --animer
```

Recharger et animer sans réentraîner :

```powershell
py -m experiences.dqn_bras --charger experiences/modeles_dqn/bras_distance_9.pt --animer
```

Options : `--actions 4`, `--mode smart`, `--mode naive`, `--angles 180 90`,
`--sans-graphes`, `--gif experiences/animations/bras_dqn.gif` et
`--sans-fenetre` (avec `--gif`).

Les fichiers `.pt` contiennent **uniquement les poids du réseau**, pas la
mémoire de replay ou l'optimiseur : ils servent à l'évaluation et à
l'animation, pas à reprendre exactement un entraînement interrompu.
Les poids générés dans `experiences/modeles_dqn/` sont ignorés par Git.
