from labyrinthe_murs import LabyrintheMurs


class LabyrintheMursFeux(LabyrintheMurs):
    """Labyrinthe qui contient a la fois des murs et des cases de feu."""

    def __init__(self, taille=3, murs=None, feux=None, penalite_feu=-1.0):
        # Exemple par defaut : mur en case 1.
        super().__init__(taille, murs=[1] if murs is None else murs)

        # Exemple par defaut : feu en case 4.
        self.feux = [4] if feux is None else list(feux)
        self.penalite_feu = penalite_feu

    def recompense(self, etat):
        """Renvoie une penalite si l'agent arrive sur une case de feu."""
        if etat in self.feux:
            return self.penalite_feu

        return super().recompense(etat)
