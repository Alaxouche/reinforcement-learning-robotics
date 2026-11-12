import numpy as np

from src.environnements.environnement_discret import EnvironnementDiscret


class QLearning:
    """Q-learning tabulaire utilisable avec tout EnvironnementDiscret."""

    INITIALISATIONS = ("aleatoire", "constante")

    def __init__(self, environnement, alpha, gamma, epsilon,
                 epsilon_min, decroissance, nb_episodes, max_pas,
                 initialisation="aleatoire", valeur_initiale=0.0):

        if not isinstance(environnement, EnvironnementDiscret):
            raise TypeError("QLearning nécessite un EnvironnementDiscret.")
        if nb_episodes <= 0:
            raise ValueError("nb_episodes doit être strictement positif.")
        if max_pas <= 0:
            raise ValueError("max_pas doit être strictement positif.")
        if initialisation not in self.INITIALISATIONS:
            raise ValueError("initialisation doit valoir 'aleatoire' ou 'constante'.")

        self.env = environnement
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decroissance = decroissance
        self.nb_episodes = nb_episodes
        self.max_pas = max_pas
        self.initialisation = initialisation

        self.recompenses = []
        self.nombre_actions = []
        self.epsilons = []  # Epsilon utilisé pour chaque épisode (avant décroissance).
        self.taux_exploration = []  # Part d'actions tirées au hasard dans chaque épisode.

        # Une ligne par état, une colonne par action.
        taille = (self.env.nb_etats, self.env.nb_actions)
        if initialisation == "aleatoire":
            # Petites valeurs initiales aléatoires (positives et négatives).
            # L'initialisation suit la graine np.random.seed(...) utilisée par
            # les expériences : les trois modes démarrent avec la même Q-table.
            self.Q = np.random.uniform(-0.1, 0.1, size=taille)
        else:
            # Même valeur partout ; aucun tirage aléatoire consommé.
            # En cas d'égalité, argmax choisit l'action de plus petit indice.
            self.Q = np.full(taille, float(valeur_initiale))

        # Compteurs cumulés : comme la table Q, ils survivent aux appels à apprendre().
        self.visites = np.zeros(self.env.nb_etats, dtype=int)
        self.somme_epsilons = np.zeros(self.env.nb_etats)  # Somme d'epsilon à chaque visite.
        self.mises_a_jour = np.zeros_like(self.Q, dtype=int)
        self.explorations = np.zeros_like(self.Q, dtype=int)  # Mises à jour dues au hasard.

    def meilleure_action(self, etat):
        """Renvoie l'action autorisée possédant la plus grande Q-value."""
        actions = list(self.env.actions_possibles(etat))
        if not actions:
            raise ValueError("Aucune action possible dans cet état.")
        indice = np.argmax(self.Q[etat, actions])
        return int(actions[indice])

    def choisir_action(self, etat):
        """Choisit une action epsilon-greedy, renvoie (action, tirée_au_hasard)."""
        actions = list(self.env.actions_possibles(etat))
        if not actions:
            raise ValueError("Aucune action possible dans cet état.")
        if np.random.random() < self.epsilon:
            return int(np.random.choice(actions)), True
        return self.meilleure_action(etat), False

    def compter_visite(self, etat):
        self.visites[etat] += 1
        self.somme_epsilons[etat] += self.epsilon

    def apprendre(self):
        """Entraine le Q-learning et collecte les indicateurs par épisode."""
        self.recompenses = []
        self.nombre_actions = []
        self.epsilons = []
        self.taux_exploration = []

        for _ in range(self.nb_episodes):
            self.epsilons.append(float(self.epsilon))
            etat = self.env.reset()
            self.compter_visite(etat)
            total = 0.0
            compteur = 0
            nb_explorations = 0
            for _ in range(self.max_pas):
                actions = self.env.actions_possibles(etat)
                if not actions:
                    break
                action, exploration = self.choisir_action(etat)
                nouvel_etat, recompense, termine = self.env.step(action)
                self.compter_visite(nouvel_etat)
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
                self.mises_a_jour[etat, action] += 1
                self.explorations[etat, action] += exploration
                nb_explorations += exploration
                total += recompense
                compteur += 1
                etat = nouvel_etat
                if termine:
                    break
            self.recompenses.append(float(total))
            self.nombre_actions.append(compteur)
            self.taux_exploration.append(nb_explorations / compteur if compteur else 0.0)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decroissance)
        return self.Q
