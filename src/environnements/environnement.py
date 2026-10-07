from abc import ABC, abstractmethod


class Environnement(ABC):
    """Classe mère la plus générale de tous les environnements.

    Un environnement, quel qu'il soit, doit au minimum savoir :
        - recommencer un épisode avec reset() ;
        - exécuter une action avec step(action).

    On ne suppose ici ni états discrets, ni actions discrètes, ni grille.
    Cela permet d'ajouter plus tard des environnements continus ou robotiques.
    """

    @abstractmethod
    def reset(self):
        """Réinitialise l'environnement et renvoie l'état initial."""
        pass

    @abstractmethod
    def step(self, action):
        """Exécute une action.

        Retour attendu :
            nouvel_etat, recompense, termine
        """
        pass
