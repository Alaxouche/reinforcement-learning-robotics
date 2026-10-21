"""Bras articule plan a deux segments, compatible avec EnvironnementDiscret.

Angles en degres : theta1 par rapport a l'horizontale ;
theta2 relatif au premier segment. Mouvement libre dans tout le plan 2D.
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
                 pas_deg=10, angles_initiaux=(90, 0), tolerance=0.15):
        self.l1, self.l2 = (float(longueurs[0]), float(longueurs[1]))
        self.cible = (float(cible[0]), float(cible[1]))
        self.pas_deg = int(pas_deg)
        self.tolerance = float(tolerance)

        if self.l1 <= 0 or self.l2 <= 0:
            raise ValueError("Les longueurs doivent etre strictement positives.")
        if (self.pas_deg <= 0 or self.pas_deg != pas_deg
                or 360 % self.pas_deg != 0):
            raise ValueError("Le pas doit etre un diviseur entier de 360.")
        if self.tolerance <= 0:
            raise ValueError("La tolerance doit etre strictement positive.")

        # Angles modulo 360 : 360 degres et 0 degre sont la meme pose.
        self.angles_theta1 = tuple(range(0, 360, self.pas_deg))
        self.angles_theta2 = self.angles_theta1
        self.nb_theta2 = len(self.angles_theta2)
        super().__init__(len(self.angles_theta1) * self.nb_theta2, 4)

        self.etat_initial = self._indices_vers_etat(
            self._indice_angle(self.angles_theta1, angles_initiaux[0]),
            self._indice_angle(self.angles_theta2, angles_initiaux[1])
        )

        # La cible peut etre dans n'importe quel quadrant, mais les
        # longueurs et le pas angulaire limitent ce qui est atteignable.
        self.etats_objectif = tuple(
            etat for etat in range(self.nb_etats)
            if self.distance_cible(etat) <= self.tolerance
        )
        if not self.etats_objectif:
            raise ValueError(
                "Cible inatteignable avec ces longueurs, le pas "
                "angulaire et la tolerance choisis."
            )

        self.etat = self.etat_initial

    @staticmethod
    def _indice_angle(angles, angle):
        angle_normalise = angle % 360
        if angle_normalise not in angles:
            raise ValueError("L'angle initial doit appartenir a la grille.")
        return angles.index(angle_normalise)

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

    def distance_cible(self, etat=None):
        _, _, extremite = self.positions(etat)
        return math.dist(extremite, self.cible)

    def reset(self):
        self.etat = self.etat_initial
        return self.etat

    def actions_possibles(self, etat):
        if not 0 <= etat < self.nb_etats:
            raise ValueError("Numero d'etat invalide.")
        if self.distance_cible(etat) <= self.tolerance:
            return []
        # Aucun mur, aucune butee : rotation circulaire dans les deux sens.
        return [0, 1, 2, 3]

    def step(self, action):
        if action not in self.actions_possibles(self.etat):
            raise ValueError("Action impossible : cible deja atteinte ou action inconnue.")
        ancienne_distance = self.distance_cible(self.etat)
        i, j = divmod(self.etat, self.nb_theta2)
        di, dj = self._deplacements[action]
        self.etat = self._indices_vers_etat(
            (i + di) % len(self.angles_theta1),
            (j + dj) % self.nb_theta2
        )
        nouvelle_distance = self.distance_cible(self.etat)
        termine = nouvelle_distance <= self.tolerance

        # Recompense dense : progres vers la cible, cout d'une action,
        # puis grand bonus lorsque l'effecteur atteint la cible.
        recompense = 2.0 * (ancienne_distance - nouvelle_distance) - 0.02
        if termine:
            recompense += 10.0
        return self.etat, float(recompense), termine
