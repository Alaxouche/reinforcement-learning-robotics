"""Bras robotique : récompense égale au rapprochement pendant le mouvement."""
from src.environnements.bras_robotique.bras_robotique_2d import BrasRobotique2D


class BrasRobotDistance(BrasRobotique2D):
    """r = d_k - d_{k+1}, sans coefficient ni bonus."""

    def _recompense_transition(self, distance_avant, distance_apres, termine):
        return distance_avant - distance_apres
