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
            etat, _, termine = bras.step(3)
        self.assertTrue(termine)
        self.assertLess(bras.distance_cible(), 1e-9)
        self.assertEqual(bras.angles(etat), (90, 270))
        self.assertEqual(bras.actions_possibles(etat), [])

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
