"""Réseau supervisé simple : position du bras -> vecteur 0/1 de l'action.
Aucune classe personnelle, aucune Q-table, aucun DQN.
"""

import numpy as np
import keras

# Les 9 actions de notre bras : variations de theta1 et theta2 (pas de 10°).
actions = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1), (0, 0),
]

# Les deux poses qui atteignent la cible (1,1), en indices de 10°.
cibles = [(0, 9), (9, 27)]


def distance(i, j, cible):
    """Nombre minimal de mouvements vers cette cible avec les 9 actions."""
    a, b = cible
    d1 = abs(i - a)
    d2 = abs(j - b)
    return max(min(d1, 36 - d1), min(d2, 36 - d2))


def direction(depart, arrivee):
    """Retourner -1, 0 ou +1 pour tourner vers l'angle voulu."""
    ecart = (arrivee - depart) % 36
    if ecart == 0:
        return 0
    return 1 if ecart <= 18 else -1


def entree(i, j):
    """Transformer deux angles en quatre nombres pour le réseau."""
    a = np.deg2rad(i * 10)
    b = np.deg2rad(j * 10)
    return [np.cos(a), np.sin(a), np.cos(b), np.sin(b)]


# 1) Fabriquer les exemples : état -> action d'un trajet optimal.
X = []
y = []
for i in range(36):
    for j in range(36):
        if (i, j) in cibles:
            continue  # Déjà sur la cible : pas d'action nécessaire.
        cible = min(cibles, key=lambda c: distance(i, j, c))
        mouvement = (direction(i, cible[0]), direction(j, cible[1]))
        X.append(entree(i, j))
        y.append(actions.index(mouvement))

X = np.array(X, dtype="float32")
Y = np.eye(9, dtype="float32")[y]  # Une seule case à 1 par exemple.

# 2) Construire et entraîner le réseau avec Keras.
reseau = keras.Sequential([
    keras.Input(shape=(4,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(9, activation="softmax"),
])
reseau.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
reseau.fit(X, Y, epochs=200, batch_size=64, verbose=0)

# 3) Tester le réseau sur une position. Modifie ces deux nombres si besoin.
theta1, theta2 = 180, 90
i, j = theta1 // 10, theta2 // 10

if (i, j) in cibles:
    print("La cible est déjà atteinte : aucune action nécessaire.")
else:
    probabilites = reseau.predict(
        np.array([entree(i, j)], dtype="float32"), verbose=0
    )[0]
    indice = int(np.argmax(probabilites))
    sortie = [0] * 9
    sortie[indice] = 1

    print("Position :", (theta1, theta2))
    print("Vecteur prédit :", sortie)
    print("Action choisie :", indice, "->", actions[indice])

# Vérification sur les mêmes positions que l'entraînement.
predictions = reseau.predict(X, verbose=0).argmax(axis=1)
print("Accord avec les exemples d'entraînement :",
      f"{100 * np.mean(predictions == np.array(y)):.1f} %")
