import numpy as np


class QLearning:
    """Agent de Q-learning tabulaire, indépendant du problème étudié.

    L'environnement doit fournir :
        nb_etats, nb_actions : dimensions de la table Q ;
        reset() : état initial au début de chaque épisode ;
        actions_possibles(etat) : liste des actions autorisées ;
        step(action) : (nouvel_etat, recompense, termine).

    'self' désigne ici l'objet QLearning : self.Q est SA matrice Q,
    et self.env pointe vers l'objet environnement reçu au constructeur.
    Les états et les actions sont représentés par des entiers commençant à 0.
    """

    def __init__(self, environnement, alpha, gamma, epsilon,
                 epsilon_min, decroissance, nb_episodes, max_pas):
        """Initialise les paramètres et une table Q remplie de zéros.

        Paramètres :
            environnement : objet avec lequel l'agent interagit.
            alpha : taux d'apprentissage (poids des nouvelles informations).
            gamma : poids accordé aux récompenses futures.
            epsilon : probabilité initiale de choisir une action au hasard.
            epsilon_min : probabilité minimale d'exploration.
            decroissance : coefficient multipliant epsilon après chaque épisode.
            nb_episodes : nombre de parties d'entraînement.
            max_pas : nombre maximal d'actions par épisode.

        Retour : aucun. Cette méthode prépare simplement l'objet QLearning.
        """

        # Paramètres de l'agent, conservés comme attributs de l'objet
        self.env = environnement  # On conserve le même objet Labyrinthe (ou autre).
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decroissance = decroissance
        self.nb_episodes = nb_episodes
        self.max_pas = max_pas

        # Table Q remplie de zéros : lignes = états, colonnes = actions
        self.Q = np.zeros((self.env.nb_etats, self.env.nb_actions))

        # Liste des scores cumulés obtenus pendant les épisodes
        self.recompenses = []

    def meilleure_action(self, etat):
        """Cherche la meilleure action autorisée selon la table Q actuelle.

        Paramètre :
            etat (int) : numéro de l'état où se trouve l'agent.

        Retour :
            int : numéro de l'action avec la plus grande Q-value.
            En cas d'égalité, la première action maximale est retenue.

        Erreur : ValueError si aucune action n'est disponible.
        """
        # Exemple pour l'état 0 d'une grille 3x3 : [1, 3] (bas, droite).
        # L'environnement décide quelles actions sont autorisées dans cet état.
        actions = list(self.env.actions_possibles(etat))
        if not actions:
            raise ValueError("Aucune action possible dans cet état.")
        # Q[etat, actions] sélectionne seulement les colonnes autorisées.
        # argmax donne la POSITION du maximum dans cette sélection, pas l'action.
        indice = np.argmax(self.Q[etat, actions])
        return int(actions[indice])  # On renvoie le numéro de l'action, pas sa Q-value.

    def choisir_action(self, etat):
        """Choisit l'action à effectuer avec la stratégie epsilon-greedy.

        Paramètre :
            etat (int) : état actuel de l'agent.

        Retour :
            int : numéro d'une action autorisée.
            Avec une probabilité epsilon, le choix est aléatoire (exploration) ;
            sinon, on prend la meilleure action connue (exploitation).

        Erreur : ValueError si aucune action n'est disponible.
        """
        actions = list(self.env.actions_possibles(etat))
        if not actions:
            raise ValueError("Aucune action possible dans cet état.")

        # Tirage entre 0 et 1 : si le nombre est inférieur à epsilon, on explore
        # Exemple : epsilon = 0.2 correspond à 20 % de tirages aléatoires.
        # Le reste du temps, on utilise la meilleure action connue.
        if np.random.random() < self.epsilon:
            return int(np.random.choice(actions))

        # Sinon, on exploite la meilleure action actuellement connue
        return self.meilleure_action(etat)

    def apprendre(self):
        """Entraîne l'agent pendant nb_episodes épisodes.

        À chaque pas : choix d'une action, interaction avec l'environnement,
        calcul de la cible de Bellman et mise à jour d'une case de la table Q.

        Retour :
            np.ndarray : la table Q obtenue après l'entraînement.

        Effets sur l'objet :
            self.Q est modifiée, self.recompenses contient le score total
            de chaque épisode et self.epsilon diminue progressivement.
        """
        self.recompenses = []  # On recommence l'historique des scores.

        # Boucle extérieure : une répétition correspond à un épisode complet.
        for episode in range(self.nb_episodes):
            etat = self.env.reset()  # Nouvel épisode : retour à l'état initial
            total = 0.0  # Somme des récompenses de cet épisode.

            # Boucle intérieure : on limite le nombre de décisions par épisode.
            for pas in range(self.max_pas):
                action = self.choisir_action(etat)  # Exploration ou exploitation.
                # L'environnement renvoie le nouvel état, la récompense et la fin éventuelle
                # Exemple de réponse : (3, -0.1, False).
                nouvel_etat, recompense, termine = self.env.step(action)

                # La valeur future est nulle si l'épisode est terminé.
                if termine:
                    valeur_future = 0.0
                else:
                    # On ne compare que les actions permises depuis le nouvel état.
                    actions_futures = list(self.env.actions_possibles(nouvel_etat))
                    if actions_futures:
                        valeur_future = np.max(self.Q[nouvel_etat, actions_futures])
                    else:
                        valeur_future = 0.0

                # Cible = récompense immédiate + gamma * meilleure valeur future.
                # La deuxième ligne corrige UNE seule case Q[etat, action].
                cible = recompense + self.gamma * valeur_future  # r + gamma * max Q(s', a')
                self.Q[etat, action] += self.alpha * (cible - self.Q[etat, action])

                total += recompense
                etat = nouvel_etat  # Le déplacement suivant partira de cet état

                # On arrête si l'épisode est fini ou s'il n'existe plus d'action
                if termine or (not self.env.actions_possibles(etat)):
                    break

            self.recompenses.append(total)  # Un score sauvegardé par épisode.
            # Exploration progressivement réduite, sans descendre sous epsilon_min
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decroissance)

        return self.Q
