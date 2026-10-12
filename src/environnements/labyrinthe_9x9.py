from src.environnements.labyrinthe import Labyrinthe


class Labyrinthe9x9(Labyrinthe):
    """Labyrinthe 9 x 9 avec murs et cases de feu.

    Disposition (S = départ, G = arrivée, # = mur, F = feu) :
        S . . # . . . . .        0  1  2  3  4  5  6  7  8
        . # . # . # # # .        9 10 11 12 13 14 15 16 17
        . # . . . F . # .       18 19 20 21 22 23 24 25 26
        . # # # # # . # .       27 28 29 30 31 32 33 34 35
        . . . F . # . . .       36 37 38 39 40 41 42 43 44
        # # # . # # # # .       45 46 47 48 49 50 51 52 53
        . F . . . . . # .       54 55 56 57 58 59 60 61 62
        . # # # # # F # .       63 64 65 66 67 68 69 70 71
        . . . . F . . . G       72 73 74 75 76 77 78 79 80

    Départ : 0
    Arrivée : 80
    Murs : 3, 10, 12, 14, 15, 16, 19, 25, 28, 29, 30, 31, 32, 34, 41,
           45, 46, 47, 49, 50, 51, 52, 61, 64, 65, 66, 67, 68, 70
    Feux : 23, 39, 55, 69, 76

    Le chemin le plus court (16 déplacements) traverse le feu ;
    le plus court sans feu fait 20 déplacements.
    """

    def __init__(self):
        super().__init__(
            taille=9,
            depart=0,
            arrivee=80,
            murs=[3, 10, 12, 14, 15, 16, 19, 25, 28, 29, 30, 31, 32, 34, 41,
                  45, 46, 47, 49, 50, 51, 52, 61, 64, 65, 66, 67, 68, 70]
        )

        self.feux = [23, 39, 55, 69, 76]

    def recompense(self, etat):
        """Renvoie la récompense propre à ce labyrinthe."""
        if etat == self.arrivee:
            return 1.0

        if etat in self.feux:
            return -1.0

        return -0.1
