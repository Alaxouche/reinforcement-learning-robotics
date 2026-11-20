# Reinforcement Learning Robotics

## Q-table définitive : la sauvegarder une seule fois

**Le réseau Keras ne génère jamais de Q-table.** Il charge uniquement le
fichier `experiences/q_tables/meilleure_q_9.npy`. S'il n'existe pas, il
s'arrête avec un message d'erreur, **sans relancer les 50 entraînements**.
Il ne modifie pas la Q-table enregistrée.

Le fichier a été produit par l'expérience précédente (50 Q-learning
indépendants, moyenne évaluée après chaque épisode, puis choix de la
meilleure table finale). Les 50 essais ne sont pas rejoués ici.

**Important :** le fichier .npy généré sur ton PC ne se trouve pas
automatiquement sur GitHub. Pour l'enregistrer dans le dépôt et pouvoir
le retrouver sur un autre ordinateur, exécuter une seule fois à la racine
du dépôt :

    git pull origin main
    git add experiences/q_tables/meilleure_q_9.npy
    git commit -m "Sauvegarder la Q-table de référence"
    git push origin main

Il est maintenant autorisé explicitement dans .gitignore, alors que
les 49 autres Q-tables et les autres fichiers .npy restent ignorés.
La sauvegarde sur GitHub est une copie distante : garder aussi une autre
copie de sauvegarde pour plus de sûreté.

## Comparaison de 4 méthodes avec Keras

On entraîne **deux réseaux** avec **la même Q-table** sauvegardée.
Puis on teste deux modes de sélection pour chaque réseau.

| Entrée | Choix de l'action |
| --- | --- |
| Deux angles theta1, theta2 | Argmax |
| Deux angles theta1, theta2 | Tirage softmax |
| Quatre valeurs cos1, sin1, cos2, sin2 | Argmax |
| Quatre valeurs cos1, sin1, cos2, sin2 | Tirage softmax |

Les réseaux utilisent 64 puis 64 neurones cachés et neuf sorties.
Les angles directs sont divisés par 360 dans la couche Rescaling.
Les cibles et les actions proviennent de notre environnement initial.

Le tirage softmax est rejoué sur tous les 1294 départs 30 fois, pour
estimer le coût moyen et sa variabilité. Pour argmax, on évalue la
politique déterministe sur les 1294 mêmes départs.
Les échecs à la limite de 50 mouvements ont un coût de 51.

Le programme sauvegarde les modèles Keras sous :
- `experiences/modeles_reseau/bras_angles.keras`
- `experiences/modeles_reseau/bras_trigo.keras`

## Lancement

Depuis la racine du projet :

    py -m pip install -r requirements.txt
    py -m experiences.reseau_bras

Modifier NB_EPOQUES_RESEAU, NB_REPETITIONS, THETA1 ou THETA2 en début
de experiences/reseau_bras.py pour changer les tests. Aucune relance
du Q-learning à l'exécution.

Les données utilisées pour entraîner les réseaux sont aussi les données
utilisées pour mesurer leurs performances. Elles ne constituent donc
pas une validation de la généralisation hors de la grille étudiée.

## Ancien Q-learning, toujours disponible séparément

    py -m experiences.suivi_apprentissage
    py -m experiences.animation_bras
