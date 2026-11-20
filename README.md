# Reinforcement Learning Robotics

## Comparer deux représentations des angles avec Keras

Le fichier experiences/reseau_bras.py compare **deux réseaux supervisés**
sur exactement la **même Q-table** et les **mêmes 1294 états** du bras.

- **Réseau A — angles** : les deux entrées sont theta1 et theta2,
  divisées par 360 par une couche Keras Rescaling.
- **Réseau B — trigo** : les quatre entrées sont cos(theta1),
  sin(theta1), cos(theta2), sin(theta2).

Les deux réseaux ont deux couches cachées Dense(64), neuf sorties
softmax, le même nombre d'époques et les mêmes actions attendues :
argmax de la meilleure Q-table à chaque état.

## D'où vient la Q-table ?

Si experiences/q_tables/meilleure_q_9.npy existe déjà sur ton ordinateur,
le programme la charge directement : **il ne relance pas 50 Q-learning**.

Sinon, il reprend notre protocole :
1. Faire 50 entraînements Q-learning indépendants (Q=0 au départ de
   chacun), 5000 épisodes chacun, 9 actions, récompense distance.
2. Après CHAQUE épisode, tester les 1294 positions de départ, sans
   exploration et sans changer Q.
3. En cas de trajet qui ne touche pas la cible en 50 mouvements, compter
   51 mouvements de pénalité.
4. À la FIN de chaque entraînement, conserver sa Q-table finale et sa
   moyenne finale.
5. Sélectionner parmi les 50 tables finales celle ayant la plus petite
   moyenne pénalisée. Sauvegarder les 50 tables et la gagnante.

## Exécuter

Depuis la racine du projet :

    py -m pip install -r requirements.txt
    py -m experiences.reseau_bras

Le programme affiche pour chaque réseau :
- Le pourcentage d'actions conformes à la Q-table ;
- Le taux de réussite sur les 1294 départs ;
- Le nombre moyen de mouvements pénalisé sur ces départs ;
- Un exemple de vecteur one-hot de neuf 0 et 1.

Il sauvegarde les modèles dans :
- experiences/modeles_reseau/bras_angles.keras
- experiences/modeles_reseau/bras_trigo.keras

Ces fichiers sont ignorés par Git et restent sur ton PC.
Les paramètres modifiables en haut du fichier :
NB_ESSAIS=50, NB_EPISODES=5000, NB_EPOQUES_RESEAU=200,
THETA1=180 et THETA2=90.

Attention : les 1294 positions d'évaluation du réseau sont aussi
les exemples d'entraînement. Le test vérifie l'imitation de la Q-table
sur la grille connue, pas la généralisation à des états inédits.
Le plus faible coût moyen mesuré servira de base à la comparaison,
mais l'avantage d'une représentation n'est pas connu avant l'exécution.

## Q-learning historique, inchangé

    py -m experiences.suivi_apprentissage
    py -m experiences.animation_bras
