from labyrinthe import Labyrinthe


class LabyrintheMurs(Labyrinthe):
    """Labyrinthe avec des murs infranchissables."""

    def __init__(self, taille=3, murs=None):
        super().__init__(taille)

        # Exemple par defaut : la case 1 est un mur.
        # On peut donner une autre liste, par exemple murs=[1, 4].
        self.murs = [1] if murs is None else list(murs)

    def etat_suivant(self, etat, action):
        """Calcule le prochain etat en empechant de traverser un mur."""
        suivant = super().etat_suivant(etat, action)

        if suivant in self.murs:
            return etat

        return suivant
