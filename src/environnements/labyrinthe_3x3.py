from src.environnements.labyrinthe import Labyrinthe


class Labyrinthe3x3(Labyrinthe):
    """Labyrinthe 3 x 3 simple, sans mur ni piège.

    Disposition :
        0  1  2
        3  4  5
        6  7  8

    Départ : 0
    Arrivée : 8
    """

    def __init__(self):
        super().__init__(
            taille=3,
            depart=0,
            arrivee=8,
            murs=[]
        )

    def recompense(self, etat):
        """Renvoie la récompense propre à ce labyrinthe."""
        if etat == self.arrivee:
            return 1.0

        return -0.1
