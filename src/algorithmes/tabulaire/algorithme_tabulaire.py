from abc import ABC

import numpy as np

from src.algorithmes.algorithme import AlgorithmeRL
from src.environnements.environnement_discret import EnvironnementDiscret


class AlgorithmeTabulaire(AlgorithmeRL, ABC):
    """Classe mère des algorithmes tabulaires.

    Elle factorise la table Q et les opérations qui sont communes aux
    algorithmes tabulaires, sans imposer la règle d'apprentissage utilisée.
    """

    def __init__(self, environnement, nb_episodes, max_pas):
        if not isinstance(environnement, EnvironnementDiscret):
            raise TypeError(
                "Un algorithme tabulaire nécessite un EnvironnementDiscret."
            )

        super().__init__(
            environnement=environnement,
            nb_episodes=nb_episodes,
            max_pas=max_pas
        )

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
        """Choisit une action selon une stratégie epsilon-greedy générique."""
        actions = list(self.env.actions_possibles(etat))

        if not actions:
            raise ValueError("Aucune action possible dans cet état.")

        if np.random.random() < epsilon:
            return int(np.random.choice(actions))

        return self.meilleure_action(etat)
