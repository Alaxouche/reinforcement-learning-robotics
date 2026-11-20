# Reinforcement Learning Robotics

Projet de bras robotique 2D. Q-learning tabulaire conservé comme référence
et réseau de neurones **supervisé en Keras** pour choisir une action.

## Installation

Depuis la racine du dépôt :

```powershell
py -m pip install -r requirements.txt
```

Keras utilise TensorFlow comme backend par défaut. Il n'est plus nécessaire
d'installer PyTorch pour ce projet.

## Fichiers utiles

- src/algorithmes/q_learning.py : Q-learning avec Q-table, conservé.
- src/algorithmes/reseau_neurones.py : réseau Keras supervisé (commenté).
- src/environnements/bras_robotique/ : bras et actions à 4 ou 9 choix.
- experiences/reseau_bras.py : entraîner, prédire, animer et évaluer.
- experiences/animation_bras.py : animation du bras réutilisée.
- experiences/suivi_apprentissage.py : graphiques du Q-learning classique.

## Réseau état s -> vecteur one-hot d'action

À chaque état du bras correspond une paire d'angles theta1 et theta2.
Les quatre entrées sont cos(theta1), sin(theta1), cos(theta2), sin(theta2),
pour respecter la périodicité des angles.

Architecture Keras :
- keras.Input(shape=(4,))
- Dense(128, activation="relu")
- Dense(128, activation="relu")
- Dense(9, activation="softmax") (ou 4 selon les actions)

Pendant l'entraînement, on calcule pour chaque état l'indice d'une action
optimale parmi les trajets les plus courts, puis on construit un vecteur
one-hot cible. Le réseau apprend avec categorical_crossentropy et Adam.

La couche softmax produit des **probabilités**. La méthode predire(s)
prend l'indice de probabilité maximale et retourne un véritable vecteur
binaire [0, 0, 1, 0, 0, 0, 0, 0, 0] si l'action d'indice 2 est choisie.

Il s'agit d'**apprentissage supervisé**, pas de DQN ni de Q-learning.
Les labels sont calculés à partir du nombre minimal théorique de mouvements
(9 actions : max des déplacements angulaires ; 4 actions : somme).
Pour un état donné, une action est optimale si elle réduit ce minimum
d'exactement un mouvement. S'il y a plusieurs bonnes actions, on prend
la première dans l'ordre de la liste des actions.

## Commandes

Entraîner sur les 1294 positions de départ et prédire l'action
pour les angles (180°,90°) :

```powershell
py -m experiences.reseau_bras --angles 180 90
```

Choisir l'état numérique ou une autre configuration :

```powershell
py -m experiences.reseau_bras --etat 657
py -m experiences.reseau_bras --actions 4 --epoques 500
```

Animer une trajectoire produite par le réseau :

```powershell
py -m experiences.reseau_bras --angles 180 90 --animer
```

Sauvegarder et recharger le modèle :

```powershell
py -m experiences.reseau_bras --sauvegarder experiences/modeles_reseau/bras_9.keras
py -m experiences.reseau_bras --charger experiences/modeles_reseau/bras_9.keras --animer
```

Les modèles sont sauvegardés au format natif .keras (et non .pt).

Après l'entraînement, on affiche la précision de classification **sur
les données d'entraînement**, le taux de réussite et le coût moyen pénalisé
sur tous les départs. Comme les mêmes positions servent à entraîner puis à
évaluer, cela ne démontre pas la généralisation à des positions inédites.

## Ancien Q-learning

```powershell
py -m experiences.suivi_apprentissage
py -m experiences.animation_bras
```