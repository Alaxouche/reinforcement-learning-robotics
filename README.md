# Reinforcement Learning Robotics

## 50 entraînements Q-learning, puis réseau Keras supervisé

Lancer depuis la racine du projet :

    py -m pip install -r requirements.txt
    py -m experiences.reseau_bras

Le fichier experiences/reseau_bras.py effectue les étapes suivantes :

1. Faire 50 essais indépendants de notre Q-learning avec 9 actions,
   récompense distance, départs aléatoires et epsilon décroissant.
   Chaque essai commence avec une nouvelle Q-table entièrement à zéro.
2. À CHAQUE épisode, tester la Q-table sur TOUS les 1294 départs
   non terminaux, sans exploration et sans changer Q.
   Calculer le nombre moyen de mouvements pour atteindre la cible.
   Un échec au-delà de 50 mouvements est pénalisé à 51.
3. À la FIN de chaque essai (5000 épisodes par défaut), garder sa Q-table
   FINALE et sa moyenne FINALE. On ne sélectionne pas un épisode
   intermédiaire au meilleur score.
4. Parmi les 50 Q-tables finales et leurs 50 moyennes, garder celle
   dont la moyenne est la plus faible.
5. Sauvegarder les 50 tables et leurs scores dans
   experiences/q_tables/50_q_tables.npz, et la gagnante seule dans
   experiences/q_tables/meilleure_q_9.npy.
6. Pour chaque état, prendre l'indice de la meilleure action de cette
   Q-table par argmax(Q[s]) comme étiquette d'entraînement.
7. Entraîner un petit réseau Keras 2 -> 64 -> 64 -> 9, à partir des deux
   angles directement (normalisation interne par 360).
8. Tester les trajectoires du réseau et afficher sa prédiction one-hot.
   Les deux graphiques représentent tous les épisodes de l'essai gagnant.

Paramètres à modifier en début de experiences/reseau_bras.py :

    NB_ESSAIS = 50
    NB_EPISODES = 5000
    NB_EPOQUES_RESEAU = 200
    THETA1, THETA2 = 180, 90

Important : 50 essais × 5000 épisodes = 250000 épisodes d'entraînement,
avec une évaluation sur tous les départs après chaque épisode.
C'est un calcul très important.

La moyenne « finale » décrit les performances à l'épisode 5000.
Un plateau visuel peut suggérer une convergence empirique, mais le
script ne garantit pas une convergence mathématique. La table gagnante
est seulement la meilleure parmi les 50 tables finales observées.

La moyenne est pénalisée tant que le taux de réussite est inférieur à
100 %. Avec 100 % de réussite, elle correspond exactement à la moyenne
des mouvements pour toucher la cible.

## Q-learning historique, conservé

    py -m experiences.suivi_apprentissage
    py -m experiences.animation_bras
