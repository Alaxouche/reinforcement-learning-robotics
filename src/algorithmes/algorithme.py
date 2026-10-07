from abc import ABC, abstractmethod

from src.environnements.environnement import Environnement


class AlgorithmeRL(ABC):
    """Classe mère générale des algorithmes d'apprentissage par renforcement.

    Elle ne suppose rien sur la représentation des états ni sur la manière
    d'apprendre. Elle stocke uniquement ce qui est commun à une expérience :
        - l'environnement ;
        - le nombre d'épisodes ;
        - le nombre maximal de pas par épisode ;
        - l'historique des récompenses.
    """

    def __init__(self, environnement, nb_episodes, max_pas):
        if not isinstance(environnement, Environnement):
            raise TypeError(
                "L'environnement doit hériter de la classe Environnement."
            )

        if nb_episodes <= 0:
            raise ValueError("nb_episodes doit être strictement positif.")
        if max_pas <= 0:
            raise ValueError("max_pas doit être strictement positif.")

        self.env = environnement
        self.nb_episodes = nb_episodes
        self.max_pas = max_pas
        self.recompenses = []

    @abstractmethod
    def choisir_action(self, etat):
        """Choisit une action à partir d'un état."""
        pass

    @abstractmethod
    def apprendre(self):
        """Entraîne l'algorithme."""
        pass
