from abc import abstractmethod

from src.environnements.environnement import Environnement


class EnvironnementDiscret(Environnement):
    """Classe mère des environnements à états et actions discrets.

    Elle ajoute seulement ce qui est nécessaire aux algorithmes tabulaires :
        - un nombre fini d'états ;
        - un nombre fini d'actions ;
        - la liste des actions possibles depuis un état.
    """

    def __init__(self, nb_etats, nb_actions):
        if nb_etats <= 0:
            raise ValueError("nb_etats doit être strictement positif.")
        if nb_actions <= 0:
            raise ValueError("nb_actions doit être strictement positif.")

        self.nb_etats = nb_etats
        self.nb_actions = nb_actions

    @abstractmethod
    def actions_possibles(self, etat):
        """Renvoie les actions autorisées depuis l'état donné."""
        pass
