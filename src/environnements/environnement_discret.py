"""Le contrat que tout environnement du projet doit respecter."""

from abc import ABC, abstractmethod


class EnvironnementDiscret(ABC):
    """Classe mere de tous les environnements a etats et actions numerotes.

    ABC signifie "Abstract Base Class" : cette classe ne se cree pas
    directement, elle sert de modele. Une classe fille qui oublie une des
    trois methodes marquees @abstractmethod ne pourra pas etre creee, et
    Python le signale tout de suite, au lieu de planter plus tard.

    Attention : Python verifie seulement que les methodes existent. Il ne
    verifie ni leurs parametres ni ce qu'elles renvoient, d'ou les precisions
    ci-dessous, qui font partie du contrat.
    """

    def __init__(self, nb_etats, nb_actions):
        if nb_etats <= 0 or nb_actions <= 0:
            raise ValueError("nb_etats et nb_actions doivent etre strictement positifs.")

        self.nb_etats = nb_etats
        self.nb_actions = nb_actions

    @abstractmethod
    def reset(self):
        """Recommence un episode. Retour : int, l'etat initial."""

    @abstractmethod
    def actions_possibles(self, etat):
        """Actions autorisees depuis cet etat. Retour : list[int].

        La liste doit etre VIDE sur un etat terminal : c'est ainsi que
        l'algorithme sait qu'il n'y a plus rien a decider.
        """

    @abstractmethod
    def step(self, action):
        """Execute une action et deplace l'environnement dans son nouvel etat.

        Parametre : action (int), prise dans actions_possibles(etat_courant).
        Retour : (nouvel_etat, recompense, termine) = (int, float, bool).
        """
