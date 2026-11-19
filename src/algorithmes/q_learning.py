"""Q-learning tabulaire : algorithme autonome, sans fonctions de graphiques."""

import numpy as np

from src.environnements.environnement_discret import EnvironnementDiscret


class QLearning:
    """Apprentissage par difference temporelle, politique epsilon-greedy."""

    def __init__(self, environnement, alpha, gamma, epsilon,
                 epsilon_min, decroissance, nb_episodes, max_pas):
        if not isinstance(environnement, EnvironnementDiscret):
            raise TypeError("QLearning necessite un EnvironnementDiscret.")
        if nb_episodes <= 0 or max_pas <= 0:
            raise ValueError("nb_episodes et max_pas doivent etre strictement positifs.")
        self.env = environnement
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decroissance = decroissance
        self.nb_episodes = nb_episodes
        self.max_pas = max_pas

        # L'entrainement commence toujours avec Q(s, a) = 0.
        self.Q = np.zeros((self.env.nb_etats, self.env.nb_actions), dtype=float)

    def meilleure_action(self, etat):
        """Meilleure action autorisee selon la Q-table actuelle."""
        actions = list(self.env.actions_possibles(etat))
        if not actions:
            raise ValueError("Aucune action possible dans cet etat.")
        return int(actions[np.argmax(self.Q[etat, actions])])

    def choisir_action(self, etat):
        """Action epsilon-greedy (exploration ou exploitation)."""
        actions = list(self.env.actions_possibles(etat))
        if not actions:
            raise ValueError("Aucune action possible dans cet etat.")
        if np.random.random() < self.epsilon:
            return int(np.random.choice(actions))
        return self.meilleure_action(etat)

    def apprendre(self):
        """Entraine la Q-table ; conserve l'acquis entre les episodes."""
        for _ in range(self.nb_episodes):
            etat = self.env.reset()
            for _ in range(self.max_pas):
                if not self.env.actions_possibles(etat):
                    break
                action = self.choisir_action(etat)
                nouvel_etat, recompense, termine = self.env.step(action)

                if termine:
                    valeur_future = 0.0
                else:
                    actions_futures = list(self.env.actions_possibles(nouvel_etat))
                    valeur_future = (
                        float(np.max(self.Q[nouvel_etat, actions_futures]))
                        if actions_futures else 0.0
                    )
                erreur_td = (
                    recompense + self.gamma * valeur_future
                    - self.Q[etat, action]
                )
                self.Q[etat, action] += self.alpha * erreur_td
                etat = nouvel_etat
                if termine:
                    break
            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.decroissance
            )
        return self.Q
