"""Bras articule plan a deux segments, compatible avec EnvironnementDiscret.

Angles en degres : theta1 par rapport a l'horizontale ;
theta2 relatif au premier segment. Le sol se trouve en y = 0.
"""

import math

from src.environnements.environnement_discret import EnvironnementDiscret


class BrasRobotique2D(EnvironnementDiscret):
    """Pilotage de deux articulations avec increments angulaires discrets."""

    noms_actions = (
        "theta1 +", "theta1 -", "theta2 +", "theta2 -"
    )
    _deplacements = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def __init__(self, longueurs=(1.0, 1.0), cible=(1.0, 1.0),
                 pas_deg=10, limites_theta1=(0, 180),
                 limites_theta2=(-150, 150),
                 angles_initiaux=(90, 0), tolerance=0.15):
        self.l1, self.l2 = (float(longueurs[0]), float(longueurs[1]))
        self.cible = (float(cible[0]), float(cible[1]))
        self.pas_deg = int(pas_deg)
        self.tolerance = float(tolerance)
        self.limites_theta1 = tuple(map(int, limites_theta1))
        self.limites_theta2 = tuple(map(int, limites_theta2))

        if self.l1 <= 0 or self.l2 <= 0:
            raise ValueError("Les longueurs doivent etre strictement positives.")
        if self.pas_deg <= 0 or self.pas_deg != pas_deg:
            raise ValueError("Le pas angulaire doit etre un entier positif.")
        if self.tolerance <= 0:
            raise ValueError("La tolerance doit etre strictement positive.")
        if self.cible[1] < 0:
            raise ValueError("La cible doit etre au-dessus du sol y = 0.")

        self.angles_theta1 = self._angles(self.limites_theta1)
        self.angles_theta2 = self._angles(self.limites_theta2)
        self.nb_theta2 = len(self.angles_theta2)
        super().__init__(len(self.angles_theta1) * self.nb_theta2, 4)

        self.etat_initial = self._indices_vers_etat(
            self._indice_angle(self.angles_theta1, angles_initiaux[0]),
            self._indice_angle(self.angles_theta2, angles_initiaux[1])
        )
        if not self.configuration_valide(self.etat_initial):
            raise ValueError("La configuration de depart traverse le sol.")
        if self.distance_cible(self.etat_initial) <= self.tolerance:
            raise ValueError("La position initiale atteint deja la cible.")

        # Une cible peut etre geometriquement atteignable mais pas avec
        # les limites articulaires et la discretisation choisies.
        self.etats_objectif = tuple(
            etat for etat in range(self.nb_etats)
            if self.configuration_valide(etat)
            and self.distance_cible(etat) <= self.tolerance
        )
        if not self.etats_objectif:
            raise ValueError(
                "Cible inatteignable avec ces longueurs, limites, "
                "pas angulaires et tolerance."
            )

        self.etat = self.etat_initial

    def _angles(self, limites):
        bas, haut = limites
        if haut <= bas or (haut - bas) % self.pas_deg:
            raise ValueError(
                "Les limites doivent etre croissantes et alignees sur le pas."
            )
        return tuple(range(bas, haut + 1, self.pas_deg))

    @staticmethod
    def _indice_angle(angles, angle):
        if angle not in angles:
            raise ValueError("L'angle initial doit appartenir a la grille.")
        return angles.index(angle)

    def _indices_vers_etat(self, indice_1, indice_2):
        return indice_1 * self.nb_theta2 + indice_2

    def angles(self, etat=None):
        """Retourne (theta1, theta2) en degres."""
        if etat is None:
            etat = self.etat
        i, j = divmod(etat, self.nb_theta2)
        return self.angles_theta1[i], self.angles_theta2[j]

    def positions(self, etat=None):
        """Retourne O, le coude A et l'effecteur M en coordonnees (x,y)."""
        theta1, theta2 = self.angles(etat)
        a1 = math.radians(theta1)
        a12 = math.radians(theta1 + theta2)
        coude = (self.l1 * math.cos(a1), self.l1 * math.sin(a1))
        extremite = (
            coude[0] + self.l2 * math.cos(a12),
            coude[1] + self.l2 * math.sin(a12)
        )
        return (0.0, 0.0), coude, extremite

    def configuration_valide(self, etat):
        """Le coude et l'effecteur doivent rester au-dessus du sol."""
        _, coude, extremite = self.positions(etat)
        return coude[1] >= -1e-9 and extremite[1] >= -1e-9

    def distance_cible(self, etat=None):
        _, _, extremite = self.positions(etat)
        return math.dist(extremite, self.cible)

    def reset(self):
        self.etat = self.etat_initial
        return self.etat

    def actions_possibles(self, etat):
        if not self.configuration_valide(etat):
            return []
        if self.distance_cible(etat) <= self.tolerance:
            return []
        i, j = divmod(etat, self.nb_theta2)
        possibles = []
        for action, (di, dj) in enumerate(self._deplacements):
            ni, nj = i + di, j + dj
            if 0 <= ni < len(self.angles_theta1) and 0 <= nj < self.nb_theta2:
                suivant = self._indices_vers_etat(ni, nj)
                if self.configuration_valide(suivant):
                    possibles.append(action)
        return possibles

    def step(self, action):
        if action not in self.actions_possibles(self.etat):
            raise ValueError("Action impossible : but atteint ou contrainte violee.")
        ancienne_distance = self.distance_cible(self.etat)
        i, j = divmod(self.etat, self.nb_theta2)
        di, dj = self._deplacements[action]
        self.etat = self._indices_vers_etat(i + di, j + dj)
        nouvelle_distance = self.distance_cible(self.etat)
        termine = nouvelle_distance <= self.tolerance

        # Recompense dense : progres vers la cible, cout d'une action,
        # puis grand bonus lorsque l'effecteur atteint la cible.
        recompense = 2.0 * (ancienne_distance - nouvelle_distance) - 0.02
        if termine:
            recompense += 10.0
        return self.etat, float(recompense), termine
