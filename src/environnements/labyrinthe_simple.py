from src.environnements.labyrinthe import Labyrinthe


class LabyrintheSimple(Labyrinthe):
    """Labyrinthe simple sans mur.

    Le départ est la première case et l'arrivée la dernière.
    """

    def __init__(self, taille=3):
        super().__init__(
            taille=taille,
            depart=0,
            arrivee=taille * taille - 1,
            murs=[]
        )
