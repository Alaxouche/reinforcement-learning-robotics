from src.environnements.labyrinthe import Labyrinthe


class Labyrinthe4x4(Labyrinthe):
    """Labyrinthe 4 x 4 utilisé pour les premiers essais du Q-learning.

    Disposition :
        12  13  14  15
         8   9  10  11
         4   5   6   7
         0   1   2   3

    Départ : 0
    Objectif : 15
    Murs : 5 et 14
    Feux : 7 et 9
    """

    def __init__(self):
        super().__init__(
            taille=4,
            depart=0,
            arrivee=15,
            murs=[5, 14]
        )

        self.feux = [7, 9]

    def recompense(self, etat):
        """Renvoie la récompense propre à ce labyrinthe."""
        if etat == self.arrivee:
            return 10.0

        if etat in self.feux:
            return -10.0

        return -0.1
