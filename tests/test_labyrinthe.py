"""Vérifications du labyrinthe 36 x 36 et de la référence optimale."""
import unittest

import numpy as np

from experiences.graphes import plus_court_chemin
from src.environnements.labyrinthe.labyrinthe_5x5 import Labyrinthe5x5
from src.environnements.labyrinthe.labyrinthe_36x36 import Labyrinthe36x36
from src.visualisation import graphes


class TestLabyrinthe36x36(unittest.TestCase):
    def test_reproductible_et_independant_de_numpy(self):
        np.random.seed(1)
        premier = Labyrinthe36x36()
        np.random.seed(2)
        second = Labyrinthe36x36()
        self.assertEqual(premier.murs, second.murs)
        self.assertEqual(premier.pieges, second.pieges)
        self.assertNotEqual(premier.murs, Labyrinthe36x36(seed=1).murs)

    def test_grille_ouverte_et_connexe(self):
        env = Labyrinthe36x36()
        self.assertEqual(env.nb_etats, 1296)
        self.assertLess(len(env.murs), 0.25 * env.nb_etats)
        self.assertNotIn(env.depart, env.murs)
        self.assertNotIn(env.arrivee, env.murs)
        self.assertFalse(set(env.pieges) & set(env.murs))
        # Toute case libre peut atteindre l'arrivée.
        optimal = graphes.retours_optimaux(env)
        for etat in range(env.nb_etats):
            if env.actions_possibles(etat):
                self.assertTrue(np.isfinite(optimal[etat]))
        self.assertGreater(plus_court_chemin(env), 34 * 2 - 1)

    def test_retours_optimaux_5x5(self):
        env = Labyrinthe5x5()
        optimal = graphes.retours_optimaux(env)
        # 0 -> 5 -> 10 -> 15 -> 16 -> 21 -> 22 -> 23 -> 24 : 7 cases à -0.1 puis +1.
        self.assertAlmostEqual(optimal[0], 1.0 - 0.7)
        self.assertAlmostEqual(optimal[23], 1.0)
        self.assertTrue(np.isnan(optimal[env.arrivee]))


if __name__ == "__main__":
    unittest.main()
