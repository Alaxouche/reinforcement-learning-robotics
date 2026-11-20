# Reinforcement Learning Robotics

Deux approches distinctes sur le même bras robotique :

- Q-learning tabulaire : src/algorithmes/q_learning.py et experiences/suivi_apprentissage.py.
- Réseau de neurones supervisé simple en Keras : experiences/reseau_bras.py.

Le réseau est indépendant du code Q-learning : aucune Q-table, récompense,
ni import de l'ancien environnement. Il reprend seulement la même grille
d'angles (10 degrés), les 9 actions et la même cible.

## Utilisation

Depuis la racine du dépôt :

    py -m pip install -r requirements.txt
    py -m experiences.reseau_bras

Modifier la ligne theta1, theta2 = 180, 90 à la fin du script pour
tester une autre position.

## Comment cela marche

1. Générer les 1294 positions qui ne touchent pas encore la cible.
2. Calculer pour chacune une action appartenant à un plus court chemin.
3. Construire les couples d'exemples (position, action one-hot).
4. Entraîner le modèle Keras Sequential avec Dense(64), Dense(64)
   et Dense(9, activation="softmax").
5. Choisir la plus grande sortie avec argmax et afficher un vecteur
   de neuf 0 et 1 avec un seul 1, par exemple [0,0,0,0,0,0,0,1,0].

Aucune classe Python personnelle, pas de DQN, pas de Q-table.
Les deux poses cibles sont (0°,90°) et (90°,270°) dans le bras initial.
Le réseau ne découvre pas l'action optimale : il imite les étiquettes
calculées. Il peut faire des erreurs.

Le pourcentage d'accord affiché est mesuré sur les états de l'entraînement :
il ne prouve pas une capacité de généralisation.

## Q-learning classique conservé

    py -m experiences.suivi_apprentissage
    py -m experiences.animation_bras
