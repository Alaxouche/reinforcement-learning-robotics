"""Deep Q-Network (DQN) pour le bras robotique discret.

Une Q-table contient une valeur pour chaque couple (état, action).
Ici, un réseau reçoit les angles et prédit les valeurs Q des actions.
L'exploration epsilon-greedy reste la même que dans notre Q-learning.

Trois mécanismes pour entraîner le réseau sans trop d'instabilité :
- mémoire d'expériences : entraînement sur des transitions mélangées ;
- réseau cible : prédictions de Bellman faites avec des poids figés ;
- perte Huber et limitation des gradients.
"""

from collections import deque
import math
from pathlib import Path
import random

import torch
from torch import nn

from src.environnements.bras_robotique.bras_robotique_2d import BrasRobotique2D


class ReseauQ(nn.Module):
    """4 entrées -> deux couches cachées de 64 neurones -> 4 ou 9 Q-valeurs."""

    def __init__(self, nb_actions, neurones=64):
        super().__init__()
        self.couches = nn.Sequential(
            nn.Linear(4, neurones),
            nn.ReLU(),
            nn.Linear(neurones, neurones),
            nn.ReLU(),
            nn.Linear(neurones, nb_actions),
        )

    def forward(self, x):
        return self.couches(x)


def encoder_angles(theta1, theta2):
    """Prendre cos et sin pour que 0° et 360° représentent la même position."""
    t1, t2 = math.radians(theta1), math.radians(theta2)
    return (math.cos(t1), math.sin(t1), math.cos(t2), math.sin(t2))


class DQN:
    """Agent DQN : Q(s,a) est prédit par le réseau et non lu dans un tableau."""

    def __init__(self, environnement, nb_episodes=5000, max_pas=50,
                 alpha=0.001, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.15, decroissance=0.9995, seed=42,
                 capacite_memoire=20000, taille_lot=64,
                 debut_apprentissage=500, entrainer_tous=4,
                 synchroniser_tous=500):
        if not isinstance(environnement, BrasRobotique2D):
            raise TypeError("Le DQN actuel attend un environnement BrasRobotique2D.")
        if nb_episodes <= 0 or max_pas <= 0:
            raise ValueError("nb_episodes et max_pas doivent être strictement positifs.")
        if not (0 <= epsilon_min <= epsilon <= 1 and 0 < decroissance <= 1):
            raise ValueError("Paramètres epsilon invalides.")
        if not (0 <= gamma <= 1 and alpha > 0):
            raise ValueError("alpha doit être positif et gamma dans [0, 1].")
        if (capacite_memoire < taille_lot or taille_lot <= 0
                or debut_apprentissage < taille_lot
                or entrainer_tous <= 0 or synchroniser_tous <= 0):
            raise ValueError("Paramètres de replay et d'entraînement invalides.")

        self.env = environnement
        self.nb_episodes = nb_episodes
        self.max_pas = max_pas
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decroissance = decroissance
        self.taille_lot = taille_lot
        self.debut_apprentissage = debut_apprentissage
        self.entrainer_tous = entrainer_tous
        self.synchroniser_tous = synchroniser_tous
        self.pas_total = 0

        # Graine reproductible et hasard indépendant des départs du bras.
        self.alea = random.Random(seed)
        torch.manual_seed(seed)

        # Pré-calcul des entrées pour les 1296 poses : angle1, angle2 -> 4 nombres.
        observations = [encoder_angles(*self.env.angles(s))
                        for s in range(self.env.nb_etats)]
        self.observations = torch.tensor(observations, dtype=torch.float32)

        self.reseau = ReseauQ(self.env.nb_actions)
        self.reseau_cible = ReseauQ(self.env.nb_actions)
        self.reseau_cible.load_state_dict(self.reseau.state_dict())
        self.reseau_cible.eval()
        self.optimiseur = torch.optim.Adam(self.reseau.parameters(), lr=alpha)
        self.perte = nn.SmoothL1Loss()  # Huber, moins sensible aux erreurs extrêmes

        # Une transition = (état, action, récompense, nouvel état, terminal).
        self.memoire = deque(maxlen=capacite_memoire)

    @torch.no_grad()
    def actions_gloutonnes(self):
        """Prédire la meilleure action pour TOUS les états en un seul calcul."""
        return self.reseau(self.observations).argmax(dim=1).numpy()

    @torch.no_grad()
    def meilleure_action(self, etat):
        """Action préférée par le réseau, sans exploration (pour les tests)."""
        actions = list(self.env.actions_possibles(etat))
        if not actions:
            raise ValueError("Aucune action disponible depuis cet état.")
        valeurs_q = self.reseau(self.observations[etat].unsqueeze(0))[0]
        return max(actions, key=lambda action: float(valeurs_q[action]))

    def choisir_action(self, etat):
        """Epsilon-greedy : exploration aléatoire ou meilleure action prédite."""
        actions = list(self.env.actions_possibles(etat))
        if self.alea.random() < self.epsilon:
            return self.alea.choice(actions)
        return self.meilleure_action(etat)

    def _mettre_a_jour_reseau(self):
        """Apprendre sur un mini-lot aléatoire de transitions mémorisées."""
        lot = self.alea.sample(self.memoire, self.taille_lot)
        etats, actions, recompenses, suivants, termines = zip(*lot)

        e = torch.tensor(etats, dtype=torch.long)
        a = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        r = torch.tensor(recompenses, dtype=torch.float32)
        s = torch.tensor(suivants, dtype=torch.long)
        finis = torch.tensor(termines, dtype=torch.bool)

        # Q du réseau principal pour les actions réellement jouées.
        q_predites = self.reseau(self.observations[e]).gather(1, a).squeeze(1)

        # Cible de Bellman : r + gamma * max Q_cible(s',a'), sans terme futur
        # si l'état suivant est terminal. Le réseau cible n'est pas optimisé ici.
        with torch.no_grad():
            q_suivantes = self.reseau_cible(self.observations[s]).max(dim=1).values
            q_cibles = r + self.gamma * q_suivantes * (~finis).float()

        erreur = self.perte(q_predites, q_cibles)
        self.optimiseur.zero_grad(set_to_none=True)
        erreur.backward()
        nn.utils.clip_grad_norm_(self.reseau.parameters(), max_norm=10.0)
        self.optimiseur.step()

    def apprendre(self, suivi_episodes=None):
        """Apprendre à partir de départs aléatoires, sans réinitialiser le réseau.

        suivi_episodes(mouvements, réussite) peut servir aux graphiques.
        """
        for _ in range(self.nb_episodes):
            etat = self.env.reset()
            mouvements, reussi = 0, False

            for _ in range(self.max_pas):
                if not self.env.actions_possibles(etat):
                    break
                action = self.choisir_action(etat)
                suivant, recompense, termine = self.env.step(action)

                # Enregistrer la transition AVANT d'ajuster le réseau.
                self.memoire.append((etat, action, recompense, suivant, termine))
                self.pas_total += 1
                mouvements += 1
                etat = suivant
                reussi = bool(termine)

                # Échantillonner le replay buffer toutes les 4 actions.
                if (len(self.memoire) >= self.debut_apprentissage
                        and self.pas_total % self.entrainer_tous == 0):
                    self._mettre_a_jour_reseau()

                # Mettre à jour les poids du réseau cible périodiquement.
                if self.pas_total % self.synchroniser_tous == 0:
                    self.reseau_cible.load_state_dict(self.reseau.state_dict())

                if termine:
                    break

            # Même décroissance epsilon par épisode que le Q-learning tabulaire.
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.decroissance)
            if suivi_episodes is not None:
                suivi_episodes(mouvements, reussi)

        return self.reseau

    def sauvegarder(self, chemin, mode):
        """Enregistrer les poids pour les réutiliser sans réentraîner."""
        chemin = Path(chemin)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"poids": self.reseau.state_dict(),
                    "nb_actions": self.env.nb_actions,
                    "mode": mode}, chemin)

    def charger(self, chemin, mode):
        """Charger un modèle compatible pour l'évaluation ou l'animation.

        La mémoire de replay et l'optimiseur ne sont PAS restaurés.
        """
        donnees = torch.load(chemin, map_location="cpu", weights_only=True)
        if donnees["nb_actions"] != self.env.nb_actions or donnees["mode"] != mode:
            raise ValueError("Le fichier ne correspond pas aux actions ou récompenses choisies.")
        self.reseau.load_state_dict(donnees["poids"])
        self.reseau_cible.load_state_dict(self.reseau.state_dict())
