"""Tests de base du bras robotique (python -m unittest discover -s tests)."""
import math
import unittest

from src.environnements.bras_robotique.bras_robotique_2d import BrasRobotique2D


class TestBrasRobotique(unittest.TestCase):
    def test_cinematique_et_cible(self):
        bras = BrasRobotique2D()
        self.assertEqual(bras.nb_etats, 589)
        self.assertEqual(bras.angles(bras.reset()), (90, 0))
        self.assertLess(math.dist(bras.positions()[-1], (0, 2)), 1e-10)
        for _ in range(9):
            etat, recompense, termine = bras.step(3)
        self.assertTrue(termine)
        self.assertLess(bras.distance_cible(), 1e-9)
        self.assertEqual(bras.angles(etat), (90, -90))
        self.assertEqual(bras.actions_possibles(etat), [])

    def test_cible_inaccessible(self):
        with self.assertRaises(ValueError):
            BrasRobotique2D(cible=(4, 1))

    def test_actions_respectent_le_sol(self):
        bras = BrasRobotique2D()
        for etat in range(bras.nb_etats):
            if not bras.configuration_valide(etat):
                self.assertEqual(bras.actions_possibles(etat), [])
            i, j = divmod(etat, bras.nb_theta2)
            for action in bras.actions_possibles(etat):
                di, dj = bras._deplacements[action]
                voisin = bras._indices_vers_etat(i + di, j + dj)
                self.assertTrue(bras.configuration_valide(voisin))


if __name__ == "__main__":
    unittest.main()
