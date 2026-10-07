import numpy as np

from src.algorithmes.tabulaire.algorithme_tabulaire import AlgorithmeTabulaire


class QLearning(AlgorithmeTabulaire):
    """Implémentation concrète de l'algorithme Q-learning tabulaire.

    QLearning ne connaît aucun labyrinthe. Il fonctionne avec n'importe quel
    EnvironnementDiscret compatible avec l'interface définie dans le projet.

    La classe mère AlgorithmeTabulaire fournit :
        - self.env ;
        - self.nb_episodes et self.max_pas ;
        - self.recompenses ;
        - self.Q ;
        - meilleure_action() ;
        - action_epsilon_greedy().
    """

    def __init__(self, environnement, alpha, gamma, epsilon,
                 epsilon_min, decroissance, nb_episodes, max_pas):
        # Initialise les éléments génériques d'un algorithme tabulaire.
        super().__init__(
            environnement=environnement,
            nb_episodes=nb_episodes,
            max_pas=max_pas
        )

        # Paramètres propres à cette implémentation du Q-learning.
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decroissance = decroissance

    def choisir_action(self, etat):
        """Choisit une action avec epsilon-greedy."""
        return self.action_epsilon_greedy(
            etat=etat,
            epsilon=self.epsilon
        )

    def apprendre(self):
        """Entraîne l'agent avec la règle de mise à jour du Q-learning."""
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

                # Règle spécifique au Q-learning :
                # Q(s,a) <- Q(s,a) + alpha * [r + gamma max Q(s',a') - Q(s,a)]
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
