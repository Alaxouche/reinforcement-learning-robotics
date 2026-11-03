"""Bras robotique : récompense avec guidage par potentiel de distance.

La forme F(s,s') = gamma * Phi(s') - Phi(s), avec Phi(s) = -distance(s)
et Phi(arrivée) = 0, fournit un signal de proximité tout en maintenant
le bonus terminal et la pénalité de pas de la récompense de base.
"""

from src.environnements.bras_robotique.bras_robotique_2d import BrasRobotique2D


class BrasRobotSmart(BrasRobotique2D):
    """r = r_base + gamma * Phi(s') - Phi(s), gamma = 0.95.

    r_base = 10 à l'arrivée, -0.02 sinon.
    Phi(s) = -distance_cible(s) hors état terminal, 0 à l'arrivée.
    GAMMA_SHAPING doit correspondre au gamma du Q-learning.
    """

    GAMMA_SHAPING = 0.95

    def _recompense_transition(self, distance_avant, distance_apres, termine):
        recompense_base = 10.0 if termine else -0.02
        potentiel_avant = -distance_avant
        potentiel_apres = 0.0 if termine else -distance_apres
        return (recompense_base
                + self.GAMMA_SHAPING * potentiel_apres
                - potentiel_avant)
