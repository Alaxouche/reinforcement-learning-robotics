# Reinforcement Learning Robotics

## Étude : imiter la meilleure Q-table avec deux réseaux Keras

La Q-table de référence est dans `experiences/q_tables/meilleure_q_9.npy`.
**Le script du réseau la charge et ne la réentraîne ni ne la modifie.**
Si elle est absente, le programme affiche une erreur plutôt que de lancer
50 nouveaux Q-learning. La Q-table doit rester sauvegardée sur GitHub.

Lancement à la racine du dépôt :

    py -m pip install -r requirements.txt
    py -m experiences.reseau_bras

### Protocole expérimental

**1. Données supervisées.** On prend les 1294 états non terminaux de notre
bras, chacun représenté par un couple d'angles. Pour chaque état, la
réponse attendue est `argmax(Q[s])` dans la Q-table sauvegardée.
On ne cherche pas à prédire les neuf valeurs de Q.

**2. Entraîner deux réseaux sur les mêmes exemples.**

- `angles` : deux entrées (theta1, theta2), normalisées par 360 dans
  une couche Keras ;
- `trigo` : quatre entrées (cos(theta1), sin(theta1),
  cos(theta2), sin(theta2)).

Les réseaux ont deux couches cachées de 64 neurones, neuf sorties
softmax, et sont entraînés pendant 200 époques avec des batchs de 64.
Les courbes précision/perte représentent les **données d'entraînement**,
pas des données de validation indépendantes.

**3. Tester deux façons de choisir une action pour chaque réseau.**

- `argmax` : prendre l'indice de la plus grande probabilité softmax ;
  évaluation déterministe sur tous les 1294 départs ;
- `tirage` : choisir aléatoirement une action selon les neuf probabilités
  softmax de l'état courant, renouveler le tirage à chaque pas ;
  refaire l'expérience sur tous les départs 30 fois.

On stoppe chaque trajectoire quand le bras atteint la cible. Après
50 mouvements sans réussite, le coût retenu est 51.

Cela donne **quatre méthodes évaluées, mais seulement deux réseaux
entraînés**. Pour les quatre, on mesure le taux de réussite et le
nombre moyen de mouvements. On compare à la Q-table d'origine.

### Quatre graphiques automatiques

En lançant le script, quatre fenêtres Matplotlib sont produites,
et quatre PNG sont enregistrés dans `experiences/resultats_reseau/` :

1. `1_precision_entrainement.png` : précision d'imitation de Q
   en fonction des époques pour les deux réseaux ;
2. `2_perte_entrainement.png` : erreur d'entraînement en fonction
   des époques ;
3. `3_taux_reussite.png` : taux de réussite des quatre méthodes
   et ligne de référence de la Q-table ;
4. `4_cout_moyen.png` : coût moyen pénalisé des quatre méthodes
   et ligne de référence de la Q-table.

Pour le tirage, la barre d'erreur du graphe 4 représente **l'écart-type
des 30 moyennes par répétition**, pas un intervalle de confiance.

### Interprétation

La métrique principale pour les trajets est le **coût moyen pénalisé**.
Le taux de réussite est indispensable pour savoir si un bon coût moyen
cache des échecs. La précision d'entraînement mesure seulement si le
réseau imite les actions recommandées par la Q-table.

Le code affiche automatiquement le meilleur réseau **observé sur cette
exécution** et la différence de coût avec la Q-table. On ne peut pas
affirmer d'avance qu'une méthode sera toujours meilleure.

Limite scientifique : les mêmes 1294 états servent à l'entraînement
et à l'évaluation ; cela mesure le comportement sur la grille connue,
pas la généralisation hors de celle-ci. Les résultats d'un seul
entraînement des réseaux ne suffisent pas à établir une supériorité
statistique générale.

### Paramètres modifiables

Dans `experiences/reseau_bras.py` :

    NB_EPOQUES_RESEAU = 200
    TAILLE_BATCH = 64
    NB_REPETITIONS = 30
    THETA1, THETA2 = 180, 90

Les deux modèles sont aussi sauvegardés dans
`experiences/modeles_reseau/bras_angles.keras` et
`experiences/modeles_reseau/bras_trigo.keras`.

## Q-learning historique

    py -m experiences.suivi_apprentissage
    py -m experiences.animation_bras
