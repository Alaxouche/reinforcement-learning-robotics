"""Bras robotique : récompense naïve."""
from src.environnements.bras_robotique.bras_robotique_2d import BrasRobotique2D


class BrasRobotNaive(BrasRobotique2D):
    """Bonus à la cible, petite pénalité à chaque autre mouvement."""

    def _recompense_transition(self, distance_avant, distance_apres, termine):
        return 10.0 if termine else -0.02
