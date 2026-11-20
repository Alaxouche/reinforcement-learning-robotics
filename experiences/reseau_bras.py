"""Réseau Keras supervisé : deux angles en entrée, une action en sortie."""

import numpy as np
import keras

# Chaque action change les angles de +10°, -10° ou 0°.
actions = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1), (0, 0)
]

# Les deux configurations qui atteignent la cible de notre bras.
cibles = [(0, 90), (90, 270)]


def cout(theta1, theta2):
    """Écart minimal à une cible, en degrés, avec rotations circulaires."""
    distances = []
    for a, b in cibles:
        d1 = abs(theta1 - a)
        d2 = abs(theta2 - b)
        distances.append(max(min(d1, 360 - d1), min(d2, 360 - d2)))
    return min(distances)


# 1. Construire les exemples : position du bras -> meilleure action.
X = []
y = []
for theta1 in range(0, 360, 10):
    for theta2 in range(0, 360, 10):
        if (theta1, theta2) in cibles:
            continue

        # Tester les 9 mouvements ; garder celui laissant le moins à parcourir.
        scores = []
        for d1, d2 in actions:
            nouveau1 = (theta1 + 10 * d1) % 360
            nouveau2 = (theta2 + 10 * d2) % 360
            scores.append(cout(nouveau1, nouveau2))

        X.append([theta1, theta2])
        y.append(np.argmin(scores))

X = np.array(X, dtype="float32")
y = np.array(y, dtype="int32")

# 2. Entraîner un réseau simple. Il reçoit directement (theta1, theta2).
reseau = keras.Sequential([
    keras.Input(shape=(2,)),
    keras.layers.Rescaling(1 / 360),  # Ramener 0..350 dans l'intervalle 0..1.
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(9, activation="softmax"),
])
reseau.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])
reseau.fit(X, y, epochs=200, batch_size=64, verbose=0)

# 3. Donner les angles au réseau et convertir son choix en vecteur de 0 et 1.
theta1, theta2 = 180, 90  # Change ces deux angles pour essayer un autre état.
if (theta1, theta2) in cibles:
    print("Le bras est déjà sur la cible.")
else:
    scores = reseau.predict(np.array([[theta1, theta2]], dtype="float32"), verbose=0)[0]
    numero = int(np.argmax(scores))
    vecteur = [0] * 9
    vecteur[numero] = 1
    print("Position :", (theta1, theta2))
    print("Vecteur 0/1 :", vecteur)
    print("Action choisie :", actions[numero])

predictions = reseau.predict(X, verbose=0).argmax(axis=1)
print("Accord sur les exemples d'entraînement :",
      f"{100*np.mean(predictions == y):.1f}%")
