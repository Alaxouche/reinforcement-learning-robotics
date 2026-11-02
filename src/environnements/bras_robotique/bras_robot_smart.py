"""Bras robotique : récompense égale à l'opposé de la distance à la cible."""
from src.environnements.bras_robotique.bras_robotique_2d import BrasRobotique2D


class BrasRobotSmart(BrasRobotique2D):
    """r = -d_{k+1}, sans autre bonus ni pénalité."""

    def _recompense_transition(self, distance_avant, distance_apres, termine):
        return -distance_apres
