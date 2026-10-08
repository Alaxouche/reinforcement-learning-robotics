import numpy as np

from src.environnements.environnement_discret import EnvironnementDiscret


class QLearning:
    """Q-learning tabulaire utilisable avec tout EnvironnementDiscret."""

    def __init__(self, environnement, alpha, gamma, epsilon,
                 epsilon_min, decroissance, nb_episodes, max_pas):

        if not isinstance(environnement, EnvironnementDiscret):
            raise TypeError(
                "QLearning nécessite un EnvironnementDiscret."
            )

        if nb_episodes <= 0:
            raise ValueError("nb_episodes doit être strictement positif.")
        if max_pas <= 0:
            raise ValueError("max_pas doit être strictement positif.")

        self.env = environnement

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decroissance = decroissance

        self.nb_episodes = nb_episodes
        self.max_pas = max_pas

        self.recompenses = []

        # Une ligne par état, une colonne par action.
        self.Q = np.zeros(
            (self.env.nb_etats, self.env.nb_actions),
            dtype=float
        )

    def meilleure_action(self, etat):
        """Renvoie l'action autorisée possédant la plus grande Q-value."""
        actions = list(self.env.actions_possibles(etat))

        if not actions:
            raise ValueError("Aucune action possible dans cet état.")

        indice = np.argmax(self.Q[etat, actions])
        return int(actions[indice])

    def choisir_action(self, etat):
        """Choisit une action avec la stratégie epsilon-greedy."""
        actions = list(self.env.actions_possibles(etat))

        if not actions:
            raise ValueError("Aucune action possible dans cet état.")

        # Exploration.
        if np.random.random() < self.epsilon:
            return int(np.random.choice(actions))

        # Exploitation.
        return self.meilleure_action(etat)

    def apprendre(self):
        """Entraîne l'agent avec la règle du Q-learning."""
        self.recompenses = []

        for episode in range(self.nb_episodes):
            etat = self.env.reset()
            total = 0.0

            for pas in range(self.max_pas):
                actions = self.env.actions_possibles(etat)

                if not actions:
                    break

                action = self.choisir_action(etat)

                nouvel_etat, recompense, termine = self.env.step(action)

                if termine:
                    valeur_future = 0.0
                else:
                    actions_futures = list(
                        self.env.actions_possibles(nouvel_etat)
                    )

                    if actions_futures:
                        valeur_future = float(
                            np.max(self.Q[nouvel_etat, actions_futures])
                        )
                    else:
                        valeur_future = 0.0

                # Mise à jour de Bellman.
                cible = recompense + self.gamma * valeur_future
                self.Q[etat, action] += self.alpha * (
                    cible - self.Q[etat, action]
                )

                total += recompense
                etat = nouvel_etat

                if termine:
                    break

            self.recompenses.append(total)

            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.decroissance
            )

        return self.Q
