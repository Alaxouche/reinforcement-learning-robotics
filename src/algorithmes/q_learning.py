import numpy as np

from src.environnements.environnement_discret import EnvironnementDiscret


class QLearning:
    """Q-learning tabulaire utilisable avec tout EnvironnementDiscret."""

    def __init__(self, environnement, alpha, gamma, epsilon,
                 epsilon_min, decroissance, nb_episodes, max_pas):

        if not isinstance(environnement, EnvironnementDiscret):
            raise TypeError("QLearning nécessite un EnvironnementDiscret.")
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
        self.nombre_actions = []
        self.variations_q = []
        self.couverture_etats = []
        self.erreurs_td = []
        self.episodes_evaluation = []
        self.recompenses_evaluation = []
        self.epsilons = []  # Epsilon utilisé pour chaque épisode (avant décroissance).

        # Une ligne par état, une colonne par action.
        self.Q = np.zeros((self.env.nb_etats, self.env.nb_actions), dtype=float)

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
        if np.random.random() < self.epsilon:
            return int(np.random.choice(actions))
        return self.meilleure_action(etat)

    def evaluer(self, nb_episodes=5):
        """Recompense moyenne de la politique gloutonne, sans mise a jour."""
        if nb_episodes <= 0:
            raise ValueError("nb_episodes doit etre strictement positif.")
        scores = []
        for _ in range(nb_episodes):
            etat = self.env.reset()
            total = 0.0
            for _ in range(self.max_pas):
                actions = self.env.actions_possibles(etat)
                if not actions:
                    break
                action = self.meilleure_action(etat)
                etat, recompense, termine = self.env.step(action)
                total += recompense
                if termine:
                    break
            scores.append(total)
        return float(np.mean(scores))

    def apprendre(self, intervalle_evaluation=None, nb_tests=5):
        """Entraine le Q-learning et collecte les indicateurs universels."""
        self.recompenses = []
        self.nombre_actions = []
        self.variations_q = []
        self.couverture_etats = []
        self.erreurs_td = []
        self.episodes_evaluation = []
        self.recompenses_evaluation = []
        self.epsilons = []

        if intervalle_evaluation is None:
            intervalle_evaluation = max(1, self.nb_episodes // 20)
        if intervalle_evaluation <= 0:
            raise ValueError("intervalle_evaluation doit etre positif.")

        etats_visites = set()

        for episode in range(self.nb_episodes):
            self.epsilons.append(float(self.epsilon))
            etat = self.env.reset()
            etats_visites.add(int(etat))
            q_precedente = self.Q.copy()
            total = 0.0
            compteur = 0
            erreurs = []
            for _ in range(self.max_pas):
                actions = self.env.actions_possibles(etat)
                if not actions:
                    break
                action = self.choisir_action(etat)
                nouvel_etat, recompense, termine = self.env.step(action)
                etats_visites.add(int(nouvel_etat))
                if termine:
                    valeur_future = 0.0
                else:
                    actions_futures = list(self.env.actions_possibles(nouvel_etat))
                    if actions_futures:
                        valeur_future = float(np.max(self.Q[nouvel_etat, actions_futures]))
                    else:
                        valeur_future = 0.0
                cible = recompense + self.gamma * valeur_future
                erreur_td = cible - self.Q[etat, action]
                self.Q[etat, action] += self.alpha * erreur_td
                erreurs.append(abs(float(erreur_td)))
                total += recompense
                compteur += 1
                etat = nouvel_etat
                if termine:
                    break
            self.recompenses.append(float(total))
            self.nombre_actions.append(compteur)
            self.variations_q.append(float(np.max(np.abs(self.Q - q_precedente))))
            self.couverture_etats.append(100.0 * len(etats_visites) / self.env.nb_etats)
            self.erreurs_td.append(float(np.mean(erreurs)) if erreurs else float("nan"))
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decroissance)
            numero = episode + 1
            if (numero == 1 or numero % intervalle_evaluation == 0
                    or numero == self.nb_episodes):
                score = self.evaluer(nb_episodes=nb_tests)
                self.episodes_evaluation.append(numero)
                self.recompenses_evaluation.append(score)
        return self.Q
