"""Réseau supervisé : état s -> action sous forme de vecteur one-hot.

Pas de Q-table ni de DQN. Les actions optimales d'entraînement sont
calculées à partir de la géométrie exacte du bras, sans BFS.
"""

import math
from pathlib import Path

import torch
from torch import nn

from src.environnements.bras_robotique.bras_robotique_2d import BrasRobotique2D


def encoder_angles(theta1, theta2):
    """Encoder les angles périodiques : 0° et 360° sont identiques."""
    a, b = math.radians(theta1), math.radians(theta2)
    return [math.cos(a), math.sin(a), math.cos(b), math.sin(b)]


def ecart_circulaire(i, j, n):
    ecart = abs(i - j)
    return min(ecart, n - ecart)


def distance_minimale(env, etat):
    """Minimum exact de mouvements depuis l'état vers une pose cible.

    9 actions : les deux articulations peuvent tourner ensemble -> max.
    4 actions : une seule articulation bouge par pas -> somme.
    """
    n1, n2 = len(env.angles_theta1), env.nb_theta2
    i, j = divmod(int(etat), n2)
    resultats = []
    for cible in env.etats_objectif:
        gi, gj = divmod(cible, n2)
        d1 = ecart_circulaire(i, gi, n1)
        d2 = ecart_circulaire(j, gj, n2)
        resultats.append(max(d1, d2) if env.nb_actions == 9 else d1 + d2)
    return min(resultats)


def action_optimale(env, etat):
    """Choisir la première action diminuant d'un pas le minimum théorique."""
    cout = distance_minimale(env, etat)
    if cout == 0:
        raise ValueError("La cible est déjà atteinte.")
    n1, n2 = len(env.angles_theta1), env.nb_theta2
    i, j = divmod(int(etat), n2)
    for indice, (di, dj) in enumerate(env.deplacements):
        suivant = ((i + di) % n1) * n2 + (j + dj) % n2
        if distance_minimale(env, suivant) == cout - 1:
            return indice
    raise RuntimeError("Aucune action diminuant le coût théorique.")


class ReseauActions(nn.Module):
    """4 entrées -> 128 neurones -> 128 neurones -> 9 (ou 4) scores."""

    def __init__(self, nb_actions):
        super().__init__()
        self.couches = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, nb_actions),
        )

    def forward(self, x):
        # Scores réels pour entraîner le réseau avec la descente de gradient.
        return self.couches(x)


class PolitiqueNeurones:
    """Utilisation : agent.predire(s) renvoie [0, ..., 1, ..., 0]."""

    def __init__(self, environnement, seed=42):
        if not isinstance(environnement, BrasRobotique2D):
            raise TypeError("Le réseau est conçu pour le bras 2D.")
        self.env = environnement
        torch.manual_seed(seed)
        self.entrees = torch.tensor(
            [encoder_angles(*self.env.angles(s))
             for s in range(self.env.nb_etats)],
            dtype=torch.float32,
        )
        self.reseau = ReseauActions(self.env.nb_actions)
        self.reseau.eval()

    def entrainer(self, epoques=800, taux=0.003):
        """Apprentissage SUPERVISÉ : enseigner une action optimale par état."""
        if epoques <= 0 or taux <= 0:
            raise ValueError("epoques et taux doivent être positifs.")
        departs = list(self.env.etats_depart)
        etiquettes = torch.tensor(
            [action_optimale(self.env, s) for s in departs],
            dtype=torch.long,
        )
        entrees = self.entrees[departs]
        optimiseur = torch.optim.Adam(self.reseau.parameters(), lr=taux)
        perte = nn.CrossEntropyLoss()
        self.reseau.train()

        for _ in range(epoques):
            scores = self.reseau(entrees)
            erreur = perte(scores, etiquettes)
            optimiseur.zero_grad()
            erreur.backward()
            optimiseur.step()

        self.reseau.eval()
        with torch.no_grad():
            pred = self.reseau(entrees).argmax(dim=1)
        return int((pred == etiquettes).sum().item()), len(departs)

    @torch.no_grad()
    def meilleure_action(self, etat):
        """Indice de l'action choisie, pour piloter aussi l'animation."""
        if not self.env.actions_possibles(int(etat)):
            raise ValueError("La cible est déjà atteinte : aucune action.")
        scores = self.reseau(self.entrees[int(etat)].unsqueeze(0))
        return int(scores.argmax(dim=1).item())

    def predire(self, etat):
        """Vecteur de 0 et de 1, avec un seul 1 à l'indice de l'action."""
        indice = self.meilleure_action(etat)
        sortie = [0] * self.env.nb_actions
        sortie[indice] = 1
        return sortie

    @torch.no_grad()
    def actions_gloutonnes(self):
        """Prédictions sur tous les états, pour l'évaluation globale."""
        return self.reseau(self.entrees).argmax(dim=1).numpy()

    def sauvegarder(self, chemin):
        chemin = Path(chemin)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"poids": self.reseau.state_dict(),
             "nb_actions": self.env.nb_actions,
             "pas_deg": self.env.pas_deg,
             "cible": tuple(self.env.cible),
             "tolerance": self.env.tolerance},
            chemin,
        )

    def charger(self, chemin):
        donnees = torch.load(chemin, map_location="cpu", weights_only=True)
        if (donnees["nb_actions"] != self.env.nb_actions
                or donnees["pas_deg"] != self.env.pas_deg
                or tuple(donnees["cible"]) != tuple(self.env.cible)
                or donnees["tolerance"] != self.env.tolerance):
            raise ValueError("Modèle incompatible avec le bras actuel.")
        self.reseau.load_state_dict(donnees["poids"])
        self.reseau.eval()
