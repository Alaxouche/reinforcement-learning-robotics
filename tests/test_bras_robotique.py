"""Vérifications du bras robotique, sans dépendance graphique."""
import unittest

import numpy as np

from src.algorithmes.q_learning import QLearning
from src.environnements.bras_robotique.bras_robot_naive import BrasRobotNaive
from src.environnements.bras_robotique.bras_robot_smart import BrasRobotSmart
from src.environnements.bras_robotique.bras_robot_distance import BrasRobotDistance
from experiences.entrainement_bras_robotique import suivre_politique


CLASSES_BRAS = {
    "naive": BrasRobotNaive,
    "smart": BrasRobotSmart,
    "distance": BrasRobotDistance,
}


class TestBrasRobotique(unittest.TestCase):
    def test_recompenses_avec_bonus_terminal(self):
        for mode in ("naive", "smart", "distance"):
            with self.subTest(mode=mode):
                env = CLASSES_BRAS[mode]()
                env.reset(etat_depart=env.etat_initial)
                distance_avant = env.distance_cible()
                nouvel_etat, recompense, termine = env.step(0)
                distance_apres = env.distance_cible(nouvel_etat)

                if mode == "naive":
                    attendu = 10.0 if termine else -0.02
                elif mode == "smart":
                    attendu = ((10.0 if termine else -0.02) + distance_avant
                               - (0.0 if termine else 0.95 * distance_apres))
                else:
                    attendu = distance_avant - distance_apres + (10.0 if termine else 0.0)
                self.assertAlmostEqual(recompense, attendu)

    def test_recompense_terminale_avec_bonus_smart_et_distance(self):
        for mode in ("smart", "distance"):
            with self.subTest(mode=mode):
                env = CLASSES_BRAS[mode]()
                # Cherche une transition menant à la cible depuis un état non terminal.
                transition = None
                for etat in env.etats_depart:
                    for action in env.actions_possibles(etat):
                        env.reset(etat_depart=etat)
                        d_avant = env.distance_cible()
                        nouvel_etat, recompense, termine = env.step(action)
                        if termine:
                            transition = (d_avant, env.distance_cible(nouvel_etat), recompense)
                            break
                    if transition is not None:
                        break
                self.assertIsNotNone(transition)
                d_avant, d_apres, recompense = transition
                attendu = (10.0 + d_avant if mode == "smart"
                           else 10.0 + d_avant - d_apres)
                self.assertAlmostEqual(recompense, attendu)

    def test_smart_guidage_potentiel_en_accord_avec_gamma(self):
        """Le guidage télescope jusqu'à la cible, avec potentiel terminal nul."""
        env = BrasRobotSmart()
        self.assertAlmostEqual(env.GAMMA_SHAPING, 0.95)
        # Pour toute transition non terminale :
        # F(s,s') = d(s) - gamma*d(s').
        self.assertAlmostEqual(
            env._recompense_transition(2.0, 1.8, False),
            -0.02 + 2.0 - 0.95 * 1.8
        )
        # À l'arrivée, Phi(s') = 0 (et non -distance résiduelle).
        self.assertAlmostEqual(
            env._recompense_transition(0.2, 0.05, True),
            10.0 + 0.2
        )

    def test_depart_aleatoire_reproductible_et_non_terminal(self):
        a = BrasRobotSmart(depart_aleatoire=True, seed=123)
        b = BrasRobotDistance(depart_aleatoire=True, seed=123)
        departs_a = [a.reset() for _ in range(100)]
        departs_b = [b.reset() for _ in range(100)]
        self.assertEqual(departs_a, departs_b)
        self.assertGreater(len(set(departs_a)), 1)
        self.assertTrue(all(etat in a.etats_depart for etat in departs_a))
        self.assertTrue(all(etat not in a.etats_objectif for etat in departs_a))

    def test_depart_fixe_et_depart_impose(self):
        fixe = BrasRobotNaive()
        self.assertEqual(fixe.reset(), fixe.etat_initial)
        env = BrasRobotNaive(depart_aleatoire=True, seed=123)
        etat_depart = env.etats_depart[10]
        self.assertEqual(env.reset(etat_depart=etat_depart), etat_depart)
        with self.assertRaises(ValueError):
            env.reset(etat_depart=-1)

    def test_entrainement_et_evaluation(self):
        np.random.seed(42)
        env = BrasRobotDistance(depart_aleatoire=True, seed=42)
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
