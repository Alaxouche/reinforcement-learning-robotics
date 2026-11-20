# Reinforcement Learning Robotics

Projet de bras robotique 2D et de labyrinthes : Q-learning tabulaire
conservé comme référence et un petit réseau de neurones supervisé pour
choisir directement une action.

## Fichiers utiles

- src/algorithmes/q_learning.py : ancien Q-learning avec Q-table.
- src/algorithmes/reseau_neurones.py : réseau de neurones état -> action.
- src/environnements/bras_robotique/ : le bras à 4 ou 9 actions.
- experiences/suivi_apprentissage.py : graphiques du Q-learning.
- experiences/animation_bras.py : animation, réutilisable avec le réseau.
- experiences/reseau_bras.py : apprentissage, prédiction et animation du réseau.

## Installation

Commande : py -m pip install -r requirements.txt

## Nouveau réseau : état s -> vecteur 0/1

Le réseau prend les angles du bras, encodés en cosinus et sinus.
Il produit des scores pour les 9 actions (ou 4), puis l'action au score
maximal est codée sous forme de vecteur one-hot, par exemple :
[0, 0, 1, 0, 0, 0, 0, 0, 0] (action d'indice 2).

Architecture : 4 entrées -> 128 -> 128 -> 9 sorties.
Pour entraîner le réseau, on calcule l'action d'un plus court trajet
possible dans le bras. Avec 9 actions, le minimum depuis une pose vers
la cible est le maximum des deux écarts angulaires minimaux ; avec
4 actions, c'est leur somme. En cas d'ex aequo, on retient la première
action optimale. Pas de BFS et pas de Q-table.

IMPORTANT : c'est de l'apprentissage SUPERVISÉ, et non du DQN. Les
bonnes actions sont fournies par la géométrie du problème ; le réseau
apprend à les reproduire. Les précisions annoncées sur ces états sont
des précisions d'entraînement, pas de généralisation.

Entraîner et afficher l'action :

    py -m experiences.reseau_bras
    py -m experiences.reseau_bras --angles 180 90
    py -m experiences.reseau_bras --etat 657
    py -m experiences.reseau_bras --actions 4 --epoques 1000

Animer le bras contrôlé par le réseau :

    py -m experiences.reseau_bras --angles 180 90 --animer

Sauvegarder, puis recharger les poids :

    py -m experiences.reseau_bras --sauvegarder experiences/modeles_reseau/bras_9.pt
    py -m experiences.reseau_bras --charger experiences/modeles_reseau/bras_9.pt

Le script calcule aussi le taux de réussite et le coût moyen global sur
tous les départs, avec pénalité 51 en cas d'échec, et le minimum exact
théorique sur la même population de départs.

## Ancien Q-learning toujours disponible

    py -m experiences.suivi_apprentissage
    py -m experiences.animation_bras
