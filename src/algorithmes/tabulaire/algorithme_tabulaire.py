from abc import ABC, abstractmethod

import numpy as np

from src.environnements.environnement_discret import EnvironnementDiscret


class AlgorithmeTabulaire(ABC):
    """Classe mère des algorithmes tabulaires sur environnement discret.

    Elle factorise tout ce qui est commun aux algorithmes comme Q-learning
    ou SARSA, sans imposer leur règle d'apprentissage.
    """

    def __init__(self, environnement, nb_episodes, max_pas):
        if not isinstance(environnement, EnvironnementDiscret):
            raise TypeError(
                "Un algorithme tabulaire nécessite un EnvironnementDiscret."
            )

        if nb_episodes <= 0:
            raise ValueError("nb_episodes doit être strictement positif.")
        if max_pas <= 0:
            raise ValueError("max_pas doit être strictement positif.")

        self.env = environnement
        self.nb_episodes = nb_episodes
        self.max_pas = max_pas
        self.recompenses = []

        # Une ligne par état et une colonne par action.
        self.Q = np.zeros(
            (self.env.nb_etats, self.env.nb_actions),
            dtype=float
        )

    def meilleure_action(self, etat):
        """Renvoie l'action autorisée dont la Q-value est maximale."""
        actions = list(self.env.actions_possibles(etat))

        if not actions:
            raise ValueError("Aucune action possible dans cet état.")

        indice = np.argmax(self.Q[etat, actions])
        return int(actions[indice])

    def action_epsilon_greedy(self, etat, epsilon):
        """Choisit une action avec la stratégie epsilon-greedy."""
        actions = list(self.env.actions_possibles(etat))

        if not actions:
            raise ValueError("Aucune action possible dans cet état.")

        if np.random.random() < epsilon:
            return int(np.random.choice(actions))

        return self.meilleure_action(etat)

    @abstractmethod
    def choisir_action(self, etat):
        """Choisit l'action utilisée par l'algorithme concret."""
        pass

    @abstractmethod
    def apprendre(self):
        """Entraîne l'algorithme concret."""
        pass
