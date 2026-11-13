"""Tests de l'animation : aucune action inventée en dehors de la politique Q."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiences.animation_bras import (
    angles_continus, animer, creer_agent, frames_interpolees,
    points_bras, position_depart, trajectoire_q,
)
from src.algorithmes.q_learning import QLearning
from src.environnements.bras_robotique.bras_robot_distance import BrasRobotDistance


class TestAnimationBras(unittest.TestCase):
    def test_cinematique_theta2_relatif(self):
        origine, coude, m = points_bras((1.0, 1.0), (90.0, -90.0))
        np.testing.assert_allclose(origine, (0.0, 0.0), atol=1e-10)
        np.testing.assert_allclose(coude, (0.0, 1.0), atol=1e-10)
        np.testing.assert_allclose(m, (1.0, 1.0), atol=1e-10)

    def test_angle_360_traverse_en_dix_degres(self):
        env = BrasRobotDistance(nb_actions=9)
        start = env._indices_vers_etat(35, 0)  # 350°, 0°
        env.reset(etat_depart=start)
        fin, _, _ = env.step(5)                # +10°, -10°
        poses = angles_continus(env, [start, fin], [5])
        self.assertEqual(poses, [(350.0, 0.0), (360.0, -10.0)])
        inter = frames_interpolees(poses, images_par_action=4, pause_finale=0)
        self.assertTrue(all(350 <= p[0][0] <= 360 for p in inter))

    def test_trajectoire_utilise_argmax_q_sans_modifier_q(self):
        env = BrasRobotDistance(nb_actions=9)
        agent = QLearning(env, alpha=.2, gamma=.95, epsilon=1.0,
                          epsilon_min=.15, decroissance=.9995,
                          nb_episodes=1, max_pas=5)
        debut = env.etats_depart[0]
        agent.Q[debut, 3] = 5.0
        avant = agent.Q.copy()
        etats, actions, reussi, raison = trajectoire_q(agent, debut, max_pas=3)
        self.assertEqual(etats[0], debut)
        self.assertEqual(actions[0], 3)
        self.assertEqual(len(etats), len(actions) + 1)
        np.testing.assert_array_equal(agent.Q, avant)
        self.assertIsInstance(reussi, bool)
        self.assertIsInstance(raison, str)

    def test_reutilise_q_table_enregistree(self):
        with TemporaryDirectory() as tmp:
            fichier = Path(tmp) / "agent.npy"
            original = creer_agent(episodes=3, enregistrer_q=fichier)
            recharge = creer_agent(episodes=3, charger_q=fichier)
            self.assertTrue(fichier.exists())
            np.testing.assert_array_equal(original.Q, recharge.Q)
            self.assertEqual(recharge.Q.shape, (1296, 9))
            self.assertEqual(position_depart(recharge.env), recharge.env.etat_initial)

    def test_gif_sauvegarde_sans_fenetre(self):
        env = BrasRobotDistance(nb_actions=4)
        agent = QLearning(env, alpha=.2, gamma=.95, epsilon=1.0,
                          epsilon_min=.15, decroissance=.9995,
                          nb_episodes=1, max_pas=5)
        s = env.etats_depart[0]
        agent.Q[s, 0] = 4.0
        etats, actions, reussi, raison = trajectoire_q(agent, s, max_pas=2)
        with TemporaryDirectory() as tmp:
            fichier = Path(tmp) / "bras.gif"
            animer(agent, etats, actions, reussi, raison, fps=5,
                   images_par_action=2, gif=fichier, afficher=False)
            self.assertGreater(fichier.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
