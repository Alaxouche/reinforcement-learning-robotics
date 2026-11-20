# Reinforcement Learning Robotics

## Comparaison de 4 méthodes avec Keras

Le programme experiences/reseau_bras.py entraîne **2 réseaux** avec
la même meilleure Q-table obtenue précédemment, puis teste **2 politiques
d'action pour chaque réseau**, donc **4 expériences** :

| Entrée du réseau | Méthode de sélection de l'action |
| --- | --- |
| 2 angles (theta1, theta2) | Argmax (action la plus probable) |
| 2 angles (theta1, theta2) | Tirage selon les probabilités softmax |
| 4 valeurs (cos1, sin1, cos2, sin2) | Argmax |
| 4 valeurs (cos1, sin1, cos2, sin2) | Tirage selon les probabilités softmax |

Les angles directs sont divisés par 360 dans une couche Keras Rescaling.
Les deux réseaux ont la même architecture cachée : 64 puis 64 neurones
et 9 sorties softmax, et les mêmes 1294 exemples de Q-table.

**Important : on n'entraîne que 2 réseaux, pas 4.**
Argmax et tirage s'appliquent aux mêmes probabilités de sortie.

## Mesures

- **Argmax** : tester la politique déterministe une fois sur les 1294 départs.
- **Tirage probabiliste** : pour chaque état courant, tirer une action
  selon les 9 probabilités du réseau. Le tirage est renouvelé à chaque pas.
  Répéter ce test sur TOUS les 1294 départs **30 fois** (par défaut).
- Pour chaque combinaison, afficher taux de réussite et nombre moyen de
  mouvements, avec échec pénalisé à 51 après 50 mouvements.
- Pour le tirage, afficher aussi l'écart-type des moyennes obtenues sur
  les 30 répétitions.
- Afficher pour chaque réseau un exemple de vecteur one-hot en sortie
  avec argmax et avec tirage.

Ces valeurs sont des résultats EMPIRIQUES sur les positions de la grille,
qui ont également servi à entraîner les réseaux. Elles ne mesurent pas
la généralisation à de nouveaux états.

## La Q-table utilisée

Si experiences/q_tables/meilleure_q_9.npy existe déjà sur ton PC,
le programme la recharge directement.

Sinon, il la génère avec notre protocole d'origine :
50 entraînements Q-learning indépendants, 5000 épisodes chacun ;
test des 1294 départs après CHAQUE épisode ; comparaison de la moyenne
FINALE de chaque Q-table ; sélection de la plus petite des 50 moyennes
finales. Échec = coût 51 après 50 mouvements.

Les 50 Q-tables et leurs scores sont enregistrés dans
experiences/q_tables/50_q_tables.npz, et la meilleure dans
experiences/q_tables/meilleure_q_9.npy.

## Utilisation

    py -m pip install -r requirements.txt
    git pull origin main
    py -m experiences.reseau_bras

Les modèles sont sauvegardés dans :
- experiences/modeles_reseau/bras_angles.keras
- experiences/modeles_reseau/bras_trigo.keras

Modifier les constantes au début de experiences/reseau_bras.py :
NB_ESSAIS, NB_EPISODES, NB_EPOQUES_RESEAU, NB_REPETITIONS, THETA1, THETA2.

## Q-learning historique (inchangé)

    py -m experiences.suivi_apprentissage
    py -m experiences.animation_bras
