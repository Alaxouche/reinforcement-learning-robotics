from src.environnements.labyrinthe import Labyrinthe


class Labyrinthe5x5(Labyrinthe):
    """Labyrinthe 5 x 5 avec murs et pièges.

    Disposition :
         0   1   2   3   4
         5   6   7   8   9
        10  11  12  13  14
        15  16  17  18  19
        20  21  22  23  24

    Départ : 0
    Arrivée : 24
    Murs : 3, 11, 13
    Pièges : 7, 17
    """

    def __init__(self):
        super().__init__(
            taille=5,
            depart=0,
            arrivee=24,
            murs=[3, 11, 13]
        )

        self.pieges = [7, 17]

    def recompense(self, etat):
        """Renvoie la récompense propre à ce labyrinthe."""
        if etat == self.arrivee:
            return 1.0

        if etat in self.pieges:
            return -1.0

        return -0.1
