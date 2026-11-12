"""Bras articule plan a deux segments, compatible avec EnvironnementDiscret.

Angles en degres : theta1 par rapport a l'horizontale ;
theta2 relatif au premier segment. Mouvement libre dans tout le plan 2D.
"""

from abc import abstractmethod
import math
import random

from src.environnements.environnement_discret import EnvironnementDiscret


class BrasRobotique2D(EnvironnementDiscret):
    """Pilotage de deux articulations avec increments angulaires discrets."""

    # 4 actions : une seule articulation bouge à chaque pas.
    # 9 actions : chaque articulation reçoit -pas, 0 ou +pas, simultanément.
    # Les quatre premières actions sont identiques dans les deux versions.
    ACTIONS = {
        4: (("theta1 +", (1, 0)), ("theta1 -", (-1, 0)),
            ("theta2 +", (0, 1)), ("theta2 -", (0, -1))),
        9: (("theta1 +", (1, 0)), ("theta1 -", (-1, 0)),
            ("theta2 +", (0, 1)), ("theta2 -", (0, -1)),
            ("theta1 +, theta2 +", (1, 1)), ("theta1 +, theta2 -", (1, -1)),
            ("theta1 -, theta2 +", (-1, 1)), ("theta1 -, theta2 -", (-1, -1)),
            ("immobile", (0, 0))),
    }

    def __init__(self, longueurs=(1.0, 1.0), cible=(1.0, 1.0),
                 pas_deg=10, angles_initiaux=(90, 0), tolerance=0.15,
                 depart_aleatoire=False, seed=None, nb_actions=4):
        self.l1, self.l2 = (float(longueurs[0]), float(longueurs[1]))
        self.cible = (float(cible[0]), float(cible[1]))
        self.pas_deg = int(pas_deg)
        self.tolerance = float(tolerance)
        self.depart_aleatoire = bool(depart_aleatoire)
        self._rng = random.Random(seed)

        if self.l1 <= 0 or self.l2 <= 0:
            raise ValueError("Les longueurs doivent etre strictement positives.")
        if (self.pas_deg <= 0 or self.pas_deg != pas_deg
                or 360 % self.pas_deg != 0):
            raise ValueError("Le pas doit etre un diviseur entier de 360.")
        if self.tolerance <= 0:
            raise ValueError("La tolerance doit etre strictement positive.")
        if nb_actions not in self.ACTIONS:
            raise ValueError("nb_actions doit valoir 4 ou 9.")
        self.noms_actions = tuple(nom for nom, _ in self.ACTIONS[nb_actions])
        self.deplacements = tuple(d for _, d in self.ACTIONS[nb_actions])

        # Angles modulo 360 : 360 degres et 0 degre sont la meme pose.
        self.angles_theta1 = tuple(range(0, 360, self.pas_deg))
        self.angles_theta2 = self.angles_theta1
        self.nb_theta2 = len(self.angles_theta2)
        super().__init__(len(self.angles_theta1) * self.nb_theta2, nb_actions)

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

        # Départs uniformes parmi les configurations qui ne touchent pas déjà la cible.
        objectifs = set(self.etats_objectif)
        self.etats_depart = tuple(
            etat for etat in range(self.nb_etats) if etat not in objectifs
        )
        if self.depart_aleatoire and not self.etats_depart:
            raise ValueError("Aucun état non terminal disponible pour les départs aléatoires.")

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

    def reset(self, etat_depart=None):
        """Choisit un départ aléatoire (si activé), ou impose un état pour tester."""
        if etat_depart is not None:
            if not isinstance(etat_depart, int) or not 0 <= etat_depart < self.nb_etats:
                raise ValueError("Etat de départ invalide.")
            self.etat = etat_depart
        elif self.depart_aleatoire:
            self.etat = self._rng.choice(self.etats_depart)
        else:
            self.etat = self.etat_initial
        return self.etat

    def actions_possibles(self, etat):
        if not 0 <= etat < self.nb_etats:
            raise ValueError("Numero d'etat invalide.")
        if self.distance_cible(etat) <= self.tolerance:
            return []
        # Aucun mur, aucune butee : rotation circulaire dans les deux sens.
        return list(range(self.nb_actions))

    @abstractmethod
    def _recompense_transition(self, distance_avant, distance_apres, termine):
        """Récompense spécifique à chaque version du bras robotique."""
        pass

    def step(self, action):
        if action not in self.actions_possibles(self.etat):
            raise ValueError("Action impossible : cible déjà atteinte ou action inconnue.")

        distance_avant = self.distance_cible()
        i, j = divmod(self.etat, self.nb_theta2)
        di, dj = self.deplacements[action]
        self.etat = self._indices_vers_etat(
            (i + di) % len(self.angles_theta1),
            (j + dj) % self.nb_theta2
        )
        distance_apres = self.distance_cible(self.etat)
        termine = distance_apres <= self.tolerance
        recompense = self._recompense_transition(
            distance_avant, distance_apres, termine
        )
        return self.etat, float(recompense), termine
