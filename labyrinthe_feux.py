from labyrinthe import Labyrinthe


class LabyrintheFeux(Labyrinthe):
    """Labyrinthe avec des cases de feu qui donnent une penalite."""

    def __init__(self, taille=3, feux=None, penalite_feu=-1.0):
        super().__init__(taille)

        # Exemple par defaut : la case 4 contient du feu.
        # On peut donner une autre liste, par exemple feux=[2, 4, 6].
        self.feux = [4] if feux is None else list(feux)
        self.penalite_feu = penalite_feu

    def recompense(self, etat):
        """Renvoie une forte penalite si l'agent arrive sur une case de feu."""
        if etat in self.feux:
            return self.penalite_feu

        return super().recompense(etat)
