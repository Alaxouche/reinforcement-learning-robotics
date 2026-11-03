"""Vérifications des indicateurs de QLearning et du chemin glouton des graphes."""
import unittest
from unittest import mock

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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

    def test_entrainement_graphes_identique_a_l_entrainement_complet(self):
        config = dict(experience.EXPERIENCES[1], nb_episodes=100, max_pas=12)
        par_script, _ = experience.entrainer(config, 42)

        np.random.seed(42)
        env = MODES_BRAS["naive"](depart_aleatoire=True, seed=42)
        complet = QLearning(environnement=env, alpha=config["alpha"], gamma=config["gamma"],
                            epsilon=1.0, epsilon_min=config["epsilon_min"],
                            decroissance=config["decroissance"], nb_episodes=100, max_pas=12)
        complet.apprendre()

        self.assertTrue(np.array_equal(par_script.Q, complet.Q))


    def test_depart_s_uniquement_pour_labyrinthe(self):
        """Les cartes du bras n'indiquent aucun départ fixe."""
        ax = mock.Mock()
        graphes.marquer_cases(ax, BrasRobotNaive(depart_aleatoire=True, seed=3))
        ax.text.assert_not_called()

        ax.reset_mock()
        graphes.marquer_cases(ax, Labyrinthe5x5())
        self.assertEqual(ax.text.call_count, 1)
        self.assertEqual(ax.text.call_args.args[2], "S")

        figure = mock.Mock()
        graphes.legende_cases(figure, BrasRobotNaive(depart_aleatoire=True),
                             "jamais visité")
        etiquettes_bras = [h.get_label()
                           for h in figure.legend.call_args.kwargs["handles"]]
        self.assertNotIn("départ", etiquettes_bras)

        figure.reset_mock()
        graphes.legende_cases(figure, Labyrinthe5x5(), "jamais visité")
        etiquettes_labyrinthe = [h.get_label()
                                for h in figure.legend.call_args.kwargs["handles"]]
        self.assertIn("départ", etiquettes_labyrinthe)

    def test_meilleure_action_textes_bras_et_fleches_labyrinthe(self):
        """Les bras affichent theta1/theta2, le labyrinthe des flèches."""
        agent = agent_bras(nb_episodes=5)
        agent.apprendre()
        with (mock.patch.object(Axes, "text", autospec=True) as texte,
              mock.patch.object(Axes, "arrow", autospec=True) as fleche,
              mock.patch.object(Figure, "savefig")):
            graphes.graphe_meilleure_action(agent, agent.env, "Bras", "inutilise.png")
            etiquettes = [appel.args[3] for appel in texte.call_args_list
                          if len(appel.args) >= 4 and isinstance(appel.args[3], str)]
            self.assertTrue(any(e in ("theta1+", "theta1-", "theta2+", "theta2-")
                                for e in etiquettes))
            self.assertNotIn("S", etiquettes)
            self.assertEqual(fleche.call_count, 0)

        np.random.seed(0)
        env = Labyrinthe5x5()
        agent_labyrinthe = QLearning(environnement=env, alpha=0.1, gamma=0.9,
                                    epsilon=1.0, epsilon_min=0.05, decroissance=0.99,
                                    nb_episodes=10, max_pas=15)
        agent_labyrinthe.apprendre()
        with (mock.patch.object(Axes, "text", autospec=True) as texte,
              mock.patch.object(Axes, "arrow", autospec=True) as fleche,
              mock.patch.object(Figure, "savefig")):
            graphes.graphe_meilleure_action(agent_labyrinthe, env,
                                            "Labyrinthe", "inutilise.png")
            self.assertGreater(fleche.call_count, 0)
            self.assertFalse(any("theta" in str(appel)
                                 for appel in texte.call_args_list))


if __name__ == "__main__":
    unittest.main()
