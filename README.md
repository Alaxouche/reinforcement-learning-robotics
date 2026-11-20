# Reinforcement Learning Robotics

## Réseau Keras supervisé (version simple)

Le fichier experiences/reseau_bras.py est indépendant du Q-learning
précédent. Il reprend seulement le même bras (deux angles, neuf actions
et la cible). Aucun code orienté objet personnel.

Installer et exécuter depuis la racine du dépôt :

    py -m pip install -r requirements.txt
    py -m experiences.reseau_bras

Pour tester un autre état, modifier simplement à la fin du fichier :

    theta1, theta2 = 180, 90

Le réseau reçoit **directement deux nombres**, les deux angles en degrés
(theta1, theta2). Aucune transformation en sinus/cosinus : Keras les divise
seulement par 360 dans une couche Rescaling interne (pour faciliter
l'apprentissage, sans changer le nombre d'entrées).

Apprentissage supervisé, en trois étapes :

1. Pour chaque position de notre grille (de 0 à 350 degrés par 10 degrés),
   tester les neuf mouvements possibles et retenir celui qui minimise
   le nombre de mouvements restant vers l'une des deux poses cibles.
   En cas d'ex aequo, on choisit la première action de la liste.
2. Entraîner un petit réseau Sequential : deux entrées, deux couches
   Dense(64) et neuf sorties softmax, avec les indices d'actions corrects.
3. Donner deux angles au réseau ; choisir l'indice avec argmax, puis afficher
   un vecteur de neuf 0 et 1 contenant un seul 1 à cet indice.

La dernière couche produit des probabilités ; le vecteur 0/1 est obtenu
après argmax. Une prédiction du réseau n'est pas garantie optimale.

Le pourcentage d'accord final est mesuré sur les mêmes positions que
l'entraînement : ce n'est pas une évaluation sur de nouvelles données.

## Ancien Q-learning, conservé séparément

    py -m experiences.suivi_apprentissage
    py -m experiences.animation_bras
