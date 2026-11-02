"""Vérifications du bras robotique, sans dépendance graphique."""
import unittest

import numpy as np

from src.algorithmes.q_learning import QLearning
from src.environnements.bras_robotique.bras_robotique_2d import BrasRobotique2D
from experiences.entrainement_bras_robotique import suivre_politique


class TestBrasRobotique(unittest.TestCase):
    def test_recompenses_ancienne_et_nouvelle(self):
        for mode in ("naive", "smart", "distance"):
            with self.subTest(mode=mode):
                env = BrasRobotique2D(mode_recompense=mode)
                env.reset(etat_depart=env.etat_initial)
                distance_avant = env.distance_cible()
                nouvel_etat, recompense, termine = env.step(0)
                distance_apres = env.distance_cible(nouvel_etat)

                if mode == "naive":
                    attendu = 10.0 if termine else -0.02
                elif mode == "smart":
                    attendu = 2.0 * (distance_avant - distance_apres) - 0.02
                    if termine:
                        attendu += 10.0
                else:
                    attendu = -distance_apres + (10.0 if termine else 0.0)
                self.assertAlmostEqual(recompense, attendu)

    def test_depart_aleatoire_reproductible_et_non_terminal(self):
        a = BrasRobotique2D(depart_aleatoire=True, seed=123, mode_recompense="smart")
        b = BrasRobotique2D(depart_aleatoire=True, seed=123, mode_recompense="distance")
        departs_a = [a.reset() for _ in range(100)]
        departs_b = [b.reset() for _ in range(100)]
        self.assertEqual(departs_a, departs_b)
        self.assertGreater(len(set(departs_a)), 1)
        self.assertTrue(all(etat in a.etats_depart for etat in departs_a))
        self.assertTrue(all(etat not in a.etats_objectif for etat in departs_a))

    def test_depart_fixe_et_depart_impose(self):
        fixe = BrasRobotique2D()
        self.assertEqual(fixe.reset(), fixe.etat_initial)
        env = BrasRobotique2D(depart_aleatoire=True, seed=123)
        etat_depart = env.etats_depart[10]
        self.assertEqual(env.reset(etat_depart=etat_depart), etat_depart)
        with self.assertRaises(ValueError):
            env.reset(etat_depart=-1)

    def test_entrainement_et_evaluation(self):
        np.random.seed(42)
        env = BrasRobotique2D(mode_recompense="distance", depart_aleatoire=True, seed=42)
        agent = QLearning(
            environnement=env, alpha=0.2, gamma=0.95,
            epsilon=1.0, epsilon_min=0.15, decroissance=0.9995,
            nb_episodes=30, max_pas=12
        )
        table = agent.apprendre()
        self.assertEqual(table.shape, (env.nb_etats, env.nb_actions))
        self.assertGreater(np.count_nonzero(table), 0)
        etats, reussi = suivre_politique(agent, env.etats_depart[0])
        self.assertGreaterEqual(len(etats), 1)
        self.assertLessEqual(len(etats), agent.max_pas + 1)
        self.assertIsInstance(reussi, bool)


if __name__ == "__main__":
    unittest.main()
