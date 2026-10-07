from abc import ABC, abstractmethod


class Environnement(ABC):
    """Classe mère abstraite de tous les environnements du projet.

    Elle définit le contrat minimal attendu par QLearning :
        - connaître le nombre d'états et d'actions ;
        - pouvoir recommencer un épisode avec reset() ;
        - indiquer les actions possibles depuis un état ;
        - exécuter une action avec step().

    Cette classe ne décrit aucun problème concret. Ce sont les classes filles
    (Labyrinthe, futur environnement robot, etc.) qui définissent leurs règles.
    """

    def __init__(self, nb_etats, nb_actions):
        """Initialise les informations communes à tous les environnements."""
        self.nb_etats = nb_etats
        self.nb_actions = nb_actions

    @abstractmethod
    def reset(self):
        """Replace l'environnement dans son état initial et renvoie cet état."""
        pass

    @abstractmethod
    def actions_possibles(self, etat):
        """Renvoie la liste des actions autorisées depuis un état."""
        pass

    @abstractmethod
    def step(self, action):
        """Exécute une action et renvoie (nouvel_etat, recompense, termine)."""
        pass
