"""Vérifications des indicateurs de QLearning et du chemin glouton des graphes."""
import unittest

import numpy as np

from experiences import graphes as experience
from experiences.entrainement_bras_robotique import MODES_BRAS
from src.algorithmes.q_learning import QLearning
from src.environnements.bras_robotique.bras_robot_naive import BrasRobotNaive
from src.environnements.labyrinthe.labyrinthe_5x5 import Labyrinthe5x5
from src.visualisation import graphes


def agent_bras(seed=42, nb_episodes=50):
    np.random.seed(seed)
    env = BrasRobotNaive(depart_aleatoire=True, seed=seed)
    return QLearning(environnement=env, alpha=0.2, gamma=0.95, epsilon=1.0,
                     epsilon_min=0.15, decroissance=0.9995,
                     nb_episodes=nb_episodes, max_pas=12)


class TestIndicateurs(unittest.TestCase):
    def test_compteurs_coherents(self):
        for env in (Labyrinthe5x5(), BrasRobotNaive(depart_aleatoire=True, seed=1)):
            with self.subTest(env=type(env).__name__):
                np.random.seed(0)
                agent = QLearning(environnement=env, alpha=0.1, gamma=0.9, epsilon=0.5,
                                  epsilon_min=0.05, decroissance=0.99,
                                  nb_episodes=40, max_pas=15)
                agent.apprendre()
                # Un passage par pas, plus un au départ de chaque épisode.
                self.assertEqual(agent.visites.sum(), agent.mises_a_jour.sum() + 40)
                self.assertEqual(agent.mises_a_jour.sum(), sum(agent.nombre_actions))
                self.assertTrue((agent.explorations <= agent.mises_a_jour).all())
                self.assertEqual(len(agent.epsilons), 40)

    def test_compteurs_cumules_entre_deux_appels(self):
        agent = agent_bras()
        agent.apprendre()
        total = agent.mises_a_jour.sum()
        agent.apprendre()
        self.assertGreater(agent.mises_a_jour.sum(), total)
        self.assertEqual(len(agent.recompenses), agent.nb_episodes)


class TestChemin(unittest.TestCase):
    def test_chemin_part_du_depart_de_reference(self):
        agent = agent_bras()
        agent.apprendre()
        for _ in range(5):
            chemin = graphes.chemin_glouton(agent, agent.env)[0]
            self.assertEqual(chemin[0], agent.env.etat_initial)

    def test_chemin_ne_change_pas_l_entrainement(self):
        temoin = agent_bras()
        temoin.apprendre()
        temoin.apprendre()

        controle = agent_bras()
        controle.apprendre()
        graphes.chemin_glouton(controle, controle.env)
        controle.apprendre()

        self.assertTrue(np.array_equal(temoin.Q, controle.Q))

    def test_blocs_identiques_a_l_entrainement_complet(self):
        config = dict(experience.EXPERIENCES[1], nb_episodes=100, max_pas=12)
        par_blocs, _, _ = experience.entrainer(config, 1.0, config["epsilon_min"], 42)

        np.random.seed(42)
        env = MODES_BRAS["naive"](depart_aleatoire=True, seed=42)
        complet = QLearning(environnement=env, alpha=config["alpha"], gamma=config["gamma"],
                            epsilon=1.0, epsilon_min=config["epsilon_min"],
                            decroissance=config["decroissance"], nb_episodes=100, max_pas=12)
        complet.apprendre()

        self.assertTrue(np.array_equal(par_blocs.Q, complet.Q))


if __name__ == "__main__":
    unittest.main()
