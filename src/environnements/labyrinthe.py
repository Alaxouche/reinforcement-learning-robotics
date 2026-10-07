from src.environnements.environnement_discret import EnvironnementDiscret


class Labyrinthe(EnvironnementDiscret):
    """Environnement concret : une grille carrée.

    Cette classe ne contient que les règles propres au labyrinthe.
    Tout ce qui est générique aux environnements discrets est hérité de
    EnvironnementDiscret.
    """

    def __init__(self, taille=3):
        self.taille = taille

        # Le nombre d'états et d'actions est géré par la classe mère.
        super().__init__(
            nb_etats=taille * taille,
            nb_actions=4
        )

        self.depart = 0
        self.arrivee = self.nb_etats - 1

        # Actions : 0=haut, 1=bas, 2=gauche, 3=droite.
        self.actions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

        self.noms_actions = [
            "Haut",
            "Bas",
            "Gauche",
            "Droite"
        ]

        self.murs = []
        self.feux = []

        self.etat = self.depart

    def reset(self):
        """Replace l'agent au départ."""
        self.etat = self.depart
        return self.etat

    def actions_possibles(self, etat):
        """Renvoie les actions autorisées depuis un état."""
        if etat == self.arrivee or etat in self.murs:
            return []

        ligne, colonne = divmod(etat, self.taille)
        possibles = []

        for action in range(self.nb_actions):
            dl, dc = self.actions[action]
            nouvelle_ligne = ligne + dl
            nouvelle_colonne = colonne + dc

            if 0 <= nouvelle_ligne < self.taille and 0 <= nouvelle_colonne < self.taille:
                suivant = nouvelle_ligne * self.taille + nouvelle_colonne

                if suivant not in self.murs:
                    possibles.append(action)

        return possibles

    def etat_suivant(self, etat, action):
        """Calcule l'état obtenu après une action."""
        if action not in self.actions_possibles(etat):
            return etat

        ligne, colonne = divmod(etat, self.taille)
        dl, dc = self.actions[action]

        return (ligne + dl) * self.taille + (colonne + dc)

    def recompense(self, etat):
        """Renvoie la récompense correspondant à l'état atteint."""
        return 1.0 if etat == self.arrivee else -0.1

    def step(self, action):
        """Exécute une action et renvoie (nouvel_etat, recompense, termine)."""
        if action not in self.actions_possibles(self.etat):
            raise ValueError("Cette action est impossible dans l'état actuel.")

        self.etat = self.etat_suivant(self.etat, action)
        recompense = self.recompense(self.etat)
        termine = self.etat == self.arrivee

        return self.etat, recompense, termine
