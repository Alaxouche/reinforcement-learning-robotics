from src.environnements.environnement_discret import EnvironnementDiscret

# Les quatre actions : 0 = Haut, 1 = Bas, 2 = Gauche, 3 = Droite.
# Chaque paire indique (variation de ligne, variation de colonne).
DEPLACEMENTS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NOMS_ACTIONS = ["Haut", "Bas", "Gauche", "Droite"]

# Recompenses, identiques pour tous les labyrinthes.
RECOMPENSE_ARRIVEE = 1.0
RECOMPENSE_FEU = -1.0
RECOMPENSE_PAS = -0.1


class Labyrinthe(EnvironnementDiscret):
    """Grille carree avec des murs et des feux.

    Regles : un mur n'est pas traversable ; un feu est accessible mais
    termine l'episode avec -1 ; l'arrivee le termine avec +1 ; tout autre
    deplacement coute -0.1.

    Les trois classes filles, en bas du fichier, ne font que choisir la
    taille et la liste des obstacles.
    """

    def __init__(self, taille=3, murs=None, feux=None):
        """Prepare la grille et verifie qu'elle est jouable.

        Parametres :
            taille (int) : cote de la grille ;
            murs (list[int]) : cases infranchissables ;
            feux (list[int]) : cases dangereuses.
        """
        super().__init__(nb_etats=taille * taille, nb_actions=len(DEPLACEMENTS))

        self.taille = taille
        
        if murs is None:
            murs = []
        if feux is None:
            feux = []

        self.murs = list(murs)
        self.feux = list(feux)

        self.depart = 0
        self.arrivee = self.nb_etats - 1
        self.etat = self.depart

        # Repris par les affichages et par la simulation.
        self.actions = DEPLACEMENTS
        self.noms_actions = NOMS_ACTIONS

        # Verifications faites une seule fois, a la creation.
        for case in self.murs + self.feux:
            if not 0 <= case < self.nb_etats:
                raise ValueError("La case {} est en dehors de la grille.".format(case))
            if case in (self.depart, self.arrivee):
                raise ValueError("Le depart et l'arrivee ne peuvent pas etre des obstacles.")

        if not self.chemin_existe():
            raise ValueError("Aucun chemin sur ne mene du depart a l'arrivee.")

    def voisine(self, etat, action):
        """Case voisine dans cette direction, ou None si mur ou bord de grille.

        Les methodes suivantes reposent toutes sur celle-ci : le calcul des
        coordonnees n'est ecrit qu'une seule fois.
        """
        ligne, colonne = divmod(etat, self.taille)
        dl, dc = DEPLACEMENTS[action]
        ligne, colonne = ligne + dl, colonne + dc

        if not (0 <= ligne < self.taille and 0 <= colonne < self.taille):
            return None  # on sortirait de la grille

        case = ligne * self.taille + colonne
        return None if case in self.murs else case

    def est_terminal(self, etat):
        """Dit si l'episode s'arrete sur cette case (arrivee ou feu)."""
        return etat == self.arrivee or etat in self.feux

    def etat_suivant(self, etat, action):
        """Case atteinte, sans deplacer le robot ; meme case si action interdite."""
        if action not in self.actions_possibles(etat):
            return etat

        return self.voisine(etat, action)

    def recompense(self, etat):
        """Recompense associee a une case. Retour : float."""
        if etat == self.arrivee:
            return RECOMPENSE_ARRIVEE
        if etat in self.feux:
            return RECOMPENSE_FEU
        return RECOMPENSE_PAS

    def chemin_existe(self):
        """Dit s'il existe un chemin evitant les murs ET les feux.

        Parcours en largeur : on explore les cases voisines de proche en proche.
        """
        a_explorer = [self.depart]
        deja_vues = {self.depart}

        while a_explorer:
            etat = a_explorer.pop(0)  # la grille est petite : pop(0) suffit

            if etat == self.arrivee:
                return True

            for action in range(self.nb_actions):
                case = self.voisine(etat, action)

                # Ni les murs (voisine renvoie None), ni les feux, ni les deja vues.
                if case is not None and case not in self.feux and case not in deja_vues:
                    deja_vues.add(case)
                    a_explorer.append(case)

        return False

    # Les trois methodes suivantes sont celles qu'impose EnvironnementDiscret.

    def reset(self):
        """Recommence un episode : le robot repart du depart. Retour : int."""
        self.etat = self.depart
        return self.etat

    def actions_possibles(self, etat):
        """Numeros des actions autorisees ; liste vide sur une case terminale.

        Exemple au coin haut gauche d'une grille vide : [1, 3] = bas, droite.
        """
        # Aucune action depuis une case terminale, ni depuis un mur
        # (le robot ne peut pas s'y trouver, mais la question peut etre posee).
        if self.est_terminal(etat) or etat in self.murs:
            return []

        return [action for action in range(self.nb_actions)
                if self.voisine(etat, action) is not None]

    def step(self, action):
        """Deplace vraiment le robot. Retour : (etat, recompense, termine)."""
        if action not in self.actions_possibles(self.etat):
            raise ValueError("Cette action est impossible dans l'etat actuel.")

        self.etat = self.etat_suivant(self.etat, action)
        return self.etat, self.recompense(self.etat), self.est_terminal(self.etat)


class Labyrinthe3x3(Labyrinthe):
    """Grille 3x3 sans obstacle : le cas le plus simple."""

    def __init__(self):
        super().__init__(taille=3)


class Labyrinthe5x5(Labyrinthe):
    """Grille 5x5 avec des murs."""

    def __init__(self):
        super().__init__(taille=5, murs=[1, 6, 7, 13, 18, 19])


class Labyrinthe9x9(Labyrinthe):
    """Grille 9x9 avec des murs et des feux."""

    def __init__(self):
        super().__init__(
            taille=9,
            murs=[1, 2, 3, 12, 21, 30, 48, 57, 66, 67, 68, 69, 70],
            feux=[14, 32, 52, 74],
        )


LABYRINTHES = {
    "3x3": Labyrinthe3x3,
    "5x5": Labyrinthe5x5,
    "9x9": Labyrinthe9x9,
}


def creer_labyrinthe(configuration="3x3"):
    """Construit le labyrinthe demande. Retour : objet Labyrinthe."""
    if configuration not in LABYRINTHES:
        raise ValueError("Configuration inconnue : {}. Choix : {}."
                         .format(configuration, ", ".join(LABYRINTHES)))

    return LABYRINTHES[configuration]()
