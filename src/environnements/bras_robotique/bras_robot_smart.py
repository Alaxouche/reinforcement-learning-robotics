"""Bras robotique : distance négative et bonus terminal de +10."""
from src.environnements.bras_robotique.bras_robotique_2d import BrasRobotique2D


class BrasRobotSmart(BrasRobotique2D):
    """r = -d_{k+1} + 10 * indicatrice(cible atteinte)."""

    def _recompense_transition(self, distance_avant, distance_apres, termine):
        return -distance_apres + (10.0 if termine else 0.0)
