from src.environnements.environnement import Environnement


class Labyrinthe(Environnement):
    """Environnement concret : une grille carrée.

    Labyrinthe hérite de la classe abstraite Environnement.
    Il fournit donc les méthodes imposées par la classe mère :
    reset(), actions_possibles() et step().
    """

    def __init__(self, taille=3):
        """Crée un labyrinthe carré de taille x taille."""
        self.taille = taille

        # La classe mère initialise les informations communes à tous les environnements.
        super().__init__(
            nb_etats=taille * taille,
            nb_actions=4
        )

        self.depart = 0
        self.arrivee = self.nb_etats - 1

        # 0=haut, 1=bas, 2=gauche, 3=droite
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.noms_actions = ["Haut", "Bas", "Gauche", "Droite"]

        # Aucun obstacle pour ce premier environnement.
        self.murs = []
        self.feux = []

        # État courant de l'environnement.
        self.etat = self.depart

    def reset(self):
        """Remet le robot au départ et renvoie l'état initial."""
        self.etat = self.depart
        return self.etat

    def actions_possibles(self, etat):
        """Renvoie les numéros des actions autorisées depuis un état."""
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
        """Calcule l'état atteint après une action."""
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
