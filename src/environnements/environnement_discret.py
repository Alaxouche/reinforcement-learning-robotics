from abc import ABC, abstractmethod


class EnvironnementDiscret(ABC):
    """Classe mère de tous les environnements discrets du projet.

    Un environnement discret possède :
        - un nombre fini d'états ;
        - un nombre fini d'actions ;
        - une méthode pour recommencer un épisode ;
        - une méthode pour connaître les actions possibles ;
        - une méthode pour exécuter une action.
    """

    def __init__(self, nb_etats, nb_actions):
        if nb_etats <= 0:
            raise ValueError("nb_etats doit être strictement positif.")
        if nb_actions <= 0:
            raise ValueError("nb_actions doit être strictement positif.")

        self.nb_etats = nb_etats
        self.nb_actions = nb_actions

    @abstractmethod
    def reset(self):
        """Réinitialise l'environnement et renvoie l'état initial."""
        pass

    @abstractmethod
    def actions_possibles(self, etat):
        """Renvoie les actions autorisées depuis l'état donné."""
        pass

    @abstractmethod
    def step(self, action):
        """Exécute une action et renvoie (nouvel_etat, recompense, termine)."""
        pass
