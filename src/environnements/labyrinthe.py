from abc import abstractmethod

from src.environnements.environnement_discret import EnvironnementDiscret


class Labyrinthe(EnvironnementDiscret):
    """Classe de base commune aux différents labyrinthes discrets.

    Les états sont numérotés de gauche à droite et de haut en bas.

    Cette classe contient uniquement la mécanique commune :
        - déplacements sur une grille carrée ;
        - calcul des actions possibles ;
        - calcul de l'état suivant ;
        - reset et step.

    La configuration et la fonction de récompense sont définies
    par les labyrinthes concrets.
    """

    def __init__(self, taille, depart, arrivee, murs):
        self.taille = taille

        super().__init__(
            nb_etats=taille * taille,
            nb_actions=4
        )

        self.depart = depart
        self.arrivee = arrivee
        self.murs = list(murs)

        # 0=haut, 1=bas, 2=gauche, 3=droite.
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

    @abstractmethod
    def recompense(self, etat):
        """Renvoie la récompense propre au labyrinthe concret."""
        pass

    def step(self, action):
        """Exécute une action et renvoie (nouvel_etat, recompense, termine)."""
        if action not in self.actions_possibles(self.etat):
            raise ValueError("Cette action est impossible dans l'état actuel.")

        self.etat = self.etat_suivant(self.etat, action)
        recompense = self.recompense(self.etat)
        termine = self.etat == self.arrivee

        return self.etat, recompense, termine
