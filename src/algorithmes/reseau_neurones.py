"""Réseau de neurones supervisé Keras pour choisir un mouvement du bras.

Entrée : un état s, représenté par les sinus/cosinus de ses deux angles.
Sortie : un vecteur one-hot : [0, ..., 1, ..., 0] pour l'action choisie.

Les exemples d'entraînement sont construits à partir des mouvements minimaux
théoriques. On n'utilise ni Q-table, ni DQN, ni récompenses pour apprendre.
"""

import math
from pathlib import Path

import keras
from keras import layers
import numpy as np

from src.environnements.bras_robotique.bras_robotique_2d import BrasRobotique2D


def encoder_angles(theta1, theta2):
    """Utiliser cos/sin pour tenir compte de la périodicité des angles."""
    a = math.radians(theta1)
    b = math.radians(theta2)
    return [math.cos(a), math.sin(a), math.cos(b), math.sin(b)]


def ecart_circulaire(i, j, n):
    """Plus petit écart entre deux indices d'angle sur un cercle."""
    ecart = abs(i - j)
    return min(ecart, n - ecart)


def distance_minimale(env, etat):
    """Nombre exact de mouvements pour rejoindre l'une des poses cibles.

    Avec 9 actions, les articulations peuvent tourner ensemble : max(d1,d2).
    Avec 4 actions, une seule tourne à la fois : d1+d2.
    Les indices angulaires sont calculés modulo 360 degrés.
    """
    n1, n2 = len(env.angles_theta1), env.nb_theta2
    i, j = divmod(int(etat), n2)
    distances = []
    for cible in env.etats_objectif:
        gi, gj = divmod(cible, n2)
        d1 = ecart_circulaire(i, gi, n1)
        d2 = ecart_circulaire(j, gj, n2)
        distances.append(max(d1, d2) if env.nb_actions == 9 else d1 + d2)
    return min(distances)


def action_optimale(env, etat):
    """Étiquette supervisée : une action diminuant le minimum d'une unité.

    En cas de plusieurs actions optimales, on choisit la première dans
    l'ordre du bras. La réponse attendue est donc déterministe.
    """
    cout = distance_minimale(env, etat)
    if cout == 0:
        raise ValueError("La cible est déjà atteinte : aucune action à choisir.")
    n1, n2 = len(env.angles_theta1), env.nb_theta2
    i, j = divmod(int(etat), n2)
    for action, (di, dj) in enumerate(env.deplacements):
        suivant = ((i + di) % n1) * n2 + (j + dj) % n2
        if distance_minimale(env, suivant) == cout - 1:
            return action
    raise RuntimeError("Aucune action diminuant le coût théorique.")


def creer_reseau(nb_actions, taux=0.003):
    """Réseau Keras classique : deux couches cachées et sortie softmax."""
    modele = keras.Sequential([
        keras.Input(shape=(4,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(nb_actions, activation="softmax"),
    ])
    modele.compile(
        optimizer=keras.optimizers.Adam(learning_rate=taux),
        loss="categorical_crossentropy",  # compare aux vecteurs one-hot
        metrics=["accuracy"],
    )
    return modele


class PolitiqueNeurones:
    """Interface : predire(s) renvoie un vecteur de zéros avec un seul 1."""

    def __init__(self, environnement, seed=42, max_pas=50):
        if not isinstance(environnement, BrasRobotique2D):
            raise TypeError("Le réseau attend un bras robotique 2D.")
        if max_pas <= 0:
            raise ValueError("max_pas doit être positif.")
        self.env = environnement
        self.max_pas = max_pas  # employé par l'évaluation globale des trajets
        keras.utils.set_random_seed(seed)

        # Une ligne par état (1296 poses avec le pas de 10°).
        self.entrees = np.asarray(
            [encoder_angles(*self.env.angles(s))
             for s in range(self.env.nb_etats)], dtype=np.float32
        )
        self.reseau = creer_reseau(self.env.nb_actions)

    def entrainer(self, epoques=300, taux=0.003):
        """Entraîner le réseau à reproduire les actions théoriques optimales."""
        if epoques <= 0 or taux <= 0:
            raise ValueError("epoques et taux doivent être positifs.")
        departs = np.asarray(self.env.etats_depart, dtype=int)
        # Pour chaque départ, identifier la meilleure action théorique.
        indices_actions = np.asarray(
            [action_optimale(self.env, int(s)) for s in departs], dtype=int
        )
        # Étiquettes dans la forme demandée, ex. [0, 0, 1, 0, ...].
        etiquettes = np.eye(self.env.nb_actions, dtype=np.float32)[indices_actions]

        # Categorical crossentropy + descente de gradient (Adam).
        self.reseau.compile(
            optimizer=keras.optimizers.Adam(learning_rate=taux),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.reseau.fit(
            self.entrees[departs], etiquettes,
            epochs=epoques, batch_size=128, shuffle=True, verbose=0,
        )
        # Précision sur les exemples utilisés pour l'entraînement (et non test).
        predictions = self.actions_gloutonnes()[departs]
        bonnes = int(np.count_nonzero(predictions == indices_actions))
        return bonnes, len(departs)

    def probabilites(self, etat):
        """9 probabilités prédites par la dernière couche softmax."""
        if not 0 <= int(etat) < self.env.nb_etats:
            raise ValueError("État invalide.")
        return self.reseau.predict(self.entrees[int(etat):int(etat) + 1], verbose=0)[0]

    def meilleure_action(self, etat):
        """Indice de l'action la plus probable, sans exploration."""
        if not self.env.actions_possibles(int(etat)):
            raise ValueError("La cible est déjà atteinte : aucune action.")
        return int(np.argmax(self.probabilites(etat)))

    def predire(self, etat):
        """Le vecteur réellement demandé : un seul 1, toutes les autres 0."""
        indice = self.meilleure_action(etat)
        resultat = [0] * self.env.nb_actions
        resultat[indice] = 1
        return resultat

    def actions_gloutonnes(self):
        """Pour tester la politique sur toutes les positions de départ."""
        scores = self.reseau.predict(self.entrees, verbose=0)
        return np.argmax(scores, axis=1)

    def sauvegarder(self, chemin):
        """Sauvegarder en format natif Keras (.keras), pas en .pt."""
        chemin = Path(chemin)
        if chemin.suffix != ".keras":
            raise ValueError("Le fichier doit se terminer par .keras")
        chemin.parent.mkdir(parents=True, exist_ok=True)
        self.reseau.save(chemin)

    def charger(self, chemin):
        """Recharger le réseau et contrôler ses dimensions d'entrée/sortie."""
        modele = keras.models.load_model(chemin)
        if modele.input_shape[-1] != 4 or modele.output_shape[-1] != self.env.nb_actions:
            raise ValueError("Modèle incompatible avec le nombre d'actions du bras.")
        self.reseau = modele
