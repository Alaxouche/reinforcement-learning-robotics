"""Tests : rotations 360 degres et cibles dans les quatre quadrants."""
import math
import unittest

from src.environnements.bras_robotique.bras_robotique_2d import BrasRobotique2D


class TestBrasRobotique(unittest.TestCase):
    def test_cinematique_et_cible(self):
        bras = BrasRobotique2D()
        self.assertEqual(bras.nb_etats, 1296)
        self.assertEqual(bras.angles(bras.reset()), (90, 0))
        self.assertLess(math.dist(bras.positions()[-1], (0, 2)), 1e-10)
        for _ in range(9):
            etat, recompense, termine = bras.step(3)
            if not termine:
                self.assertAlmostEqual(recompense, -0.02)
        self.assertTrue(termine)
        self.assertAlmostEqual(recompense, 10.0)
        self.assertLess(bras.distance_cible(), 1e-9)
        self.assertEqual(bras.angles(etat), (90, 270))
        self.assertEqual(bras.actions_possibles(etat), [])

    def test_recompenses_naive_et_smart(self):
        # Vers la meme cible, le mouvement theta2 - 10 rapproche le bras.
        naive = BrasRobotique2D(mode_recompense="naive")
        smart = BrasRobotique2D(mode_recompense="smart")
        avant = smart.distance_cible()
        _, recompense_naive, termine_naive = naive.step(3)
        _, recompense_smart, termine_smart = smart.step(3)
        self.assertFalse(termine_naive)
        self.assertFalse(termine_smart)
        self.assertAlmostEqual(recompense_naive, -0.02)
        self.assertAlmostEqual(
            recompense_smart, avant - smart.distance_cible() - 0.02
        )
        self.assertGreater(recompense_smart, recompense_naive)

        # Au bout de neuf pas vers la cible, le bonus final est le meme.
        for mode in ("naive", "smart"):
            with self.subTest(mode=mode):
                robot = BrasRobotique2D(mode_recompense=mode)
                for _ in range(9):
                    _, r, fini = robot.step(3)
                self.assertTrue(fini)
                self.assertAlmostEqual(r, 10.0)

        with self.assertRaises(ValueError):
            BrasRobotique2D(mode_recompense="inconnu")

    def test_rotation_circulaire_des_deux_angles(self):
        bras = BrasRobotique2D(angles_initiaux=(0, 0))
        bras.step(1)
        self.assertEqual(bras.angles(), (350, 0))
        bras.step(0)
        self.assertEqual(bras.angles(), (0, 0))
        bras.step(3)
        self.assertEqual(bras.angles(), (0, 350))
        bras.step(2)
        self.assertEqual(bras.angles(), (0, 0))
        # Les angles peuvent etre donnes en dehors de [0, 360).
        autre = BrasRobotique2D(angles_initiaux=(450, -360))
        self.assertEqual(autre.angles(), (90, 0))

    def test_cibles_quatre_quadrants(self):
        for cible in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            with self.subTest(cible=cible):
                bras = BrasRobotique2D(cible=cible)
                self.assertTrue(bras.etats_objectif)
                self.assertLess(
                    min(bras.distance_cible(s) for s in bras.etats_objectif),
                    1e-9
                )

    def test_cible_inatteignable(self):
        with self.assertRaises(ValueError):
            BrasRobotique2D(cible=(4, -1))
        with self.assertRaises(ValueError):
            BrasRobotique2D(pas_deg=7)

    def test_positions_sous_zero_autorisees(self):
        bras = BrasRobotique2D(angles_initiaux=(180, 90))
        _, coude, extremite = bras.positions()
        self.assertLess(coude[0], 0)
        self.assertLess(extremite[1], 0)
        self.assertEqual(bras.actions_possibles(bras.etat), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
