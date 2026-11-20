# Reinforcement Learning Robotics

## Q-learning puis réseau Keras supervisé

Le script experiences/reseau_bras.py n'utilise plus le calcul géométrique
des actions optimales. Il récupère ses étiquettes de notre Q-learning.

1. Q-learning original pendant 5000 épisodes : 9 actions, récompense distance,
   départs aléatoires, epsilon décroissant.
2. Après CHAQUE épisode, tester la Q-table sur TOUS les 1294 départs,
   sans epsilon et sans la modifier : réussite dans les 50 mouvements,
   échec pénalisé à 51 mouvements.
3. Conserver une COPIE de la Q-table dont le coût moyen pénalisé est
   le plus faible. À égalité, préférer le plus grand nombre de réussites.
4. Sauvegarder cette table dans experiences/q_tables/meilleure_q_9.npy.
5. Transformer chaque état en deux angles et obtenir sa réponse par
   argmax des neuf valeurs de la meilleure Q-table.
6. Entraîner le réseau Keras 2 -> 64 -> 64 -> 9 à imiter ces réponses.
   Les deux angles sont transmis directement, avec normalisation par 360.
7. Tester les trajectoires du réseau sur tous les départs, afficher une
   sortie one-hot et les deux graphiques de notre Q-learning.

Installation depuis la racine du projet :

    py -m pip install -r requirements.txt

Exécution :

    py -m experiences.reseau_bras

Modifier NB_EPISODES, NB_EPOQUES_RESEAU, THETA1 et THETA2 en haut du
fichier pour changer les paramètres.

Attention : on choisit la meilleure Q-table OBSERVÉE pendant l'entraînement,
pas nécessairement une table théoriquement optimale. Un réseau entraîné
à l'imiter peut faire des erreurs ; sa politique est donc évaluée
séparément. Les états d'entraînement servent aussi à l'évaluation,
ce qui ne prouve pas la généralisation à de nouveaux états.

## Q-learning classique conservé

    py -m experiences.suivi_apprentissage
    py -m experiences.animation_bras
