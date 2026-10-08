from src.environnements.labyrinthe import Labyrinthe


class LabyrintheSimple(Labyrinthe):
    """Labyrinthe simple sans mur.

    Le départ est la première case et l'arrivée la dernière.
    Chaque déplacement coûte -0.1 et atteindre l'arrivée rapporte +1.
    """

    def __init__(self, taille=3):
        super().__init__(
            taille=taille,
            depart=0,
            arrivee=taille * taille - 1,
            murs=[]
        )

    def recompense(self, etat):
        """Définit les récompenses propres à ce labyrinthe."""
        return 1.0 if etat == self.arrivee else -0.1
