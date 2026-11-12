"""Vérifications des indicateurs de QLearning et du chemin glouton des graphes."""
import unittest
from unittest import mock

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, to_rgba

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
    def test_initialisation_aleatoire_reproductible_de_q(self):
        """Q est aléatoire, centrée sur zéro et reproductible par graine."""
        config = experience.EXPERIENCES_BRAS[0]
        premier, _ = experience.creer_agent(config, 42)
        second, _ = experience.creer_agent(config, 42)
        autre, _ = experience.creer_agent(config, 43)

        np.testing.assert_array_equal(premier.Q, second.Q)
        self.assertFalse(np.array_equal(premier.Q, autre.Q))
        self.assertEqual(premier.Q.shape, (1296, 4))
        self.assertTrue((premier.Q >= -0.1).all())
        self.assertTrue((premier.Q <= 0.1).all())
        self.assertTrue((premier.Q < 0).any())
        self.assertTrue((premier.Q > 0).any())
        self.assertFalse((premier.Q == 0).all())

        # Pour comparer les récompenses, leur Q-table initiale est identique
        # (à nombre d'actions égal ; 9 actions donnent une table 1296 x 9).
        references = {}
        for config_mode in experience.EXPERIENCES_BRAS:
            agent, _ = experience.creer_agent(config_mode, 42)
            self.assertEqual(agent.Q.shape, (1296, config_mode["nb_actions"]))
            reference = references.setdefault(config_mode["nb_actions"], agent.Q)
            np.testing.assert_array_equal(reference, agent.Q)
        np.testing.assert_array_equal(premier.Q, references[4])
        self.assertEqual(sorted(references), [4, 9])

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

    def test_q_values_bleus_sans_changer_les_valeurs(self):
        """Quatre cartes bleues à échelle commune ; données absentes grises."""
        env = Labyrinthe5x5()
        agent = QLearning(environnement=env, alpha=0.1, gamma=0.9,
                          epsilon=1.0, epsilon_min=0.05, decroissance=0.99,
                          nb_episodes=1, max_pas=2)
        agent.Q[0, 0] = -3.0
        agent.Q[1, 0] = 5.0
        agent.mises_a_jour[0, 0] = 1
        agent.mises_a_jour[1, 0] = 1

        images = []
        original_imshow = Axes.imshow

        def enregistrer_image(ax, *args, **kwargs):
            image = original_imshow(ax, *args, **kwargs)
            images.append(image)
            return image

        with (mock.patch.object(Axes, "imshow", enregistrer_image),
              mock.patch.object(Figure, "savefig")):
            graphes.graphe_cartes_q_brutes(agent, env, "Labyrinthe", "inutilise.png")

        self.assertEqual(len(images), env.nb_actions)
        for image in images:
            self.assertEqual(image.norm.vmin, -3.0)
            self.assertEqual(image.norm.vmax, 5.0)
            self.assertEqual(image.cmap.name, graphes.BLEUS_Q.name)
            self.assertTrue(np.allclose(image.cmap.get_bad(),
                                        to_rgba(graphes.SANS_DONNEE)))
        self.assertAlmostEqual(float(images[0].get_array()[0, 0]), -3.0)
        self.assertAlmostEqual(float(images[0].get_array()[0, 1]), 5.0)
        self.assertTrue(np.ma.getmaskarray(images[0].get_array())[0, 2])

    def test_cartes_comptage_echelle_log(self):
        """Les cartes utilisent LogNorm et laissent les zéros en gris."""
        environnements = [Labyrinthe5x5(),
                          BrasRobotNaive(depart_aleatoire=True, seed=42)]
        for env in environnements:
            with self.subTest(env=type(env).__name__):
                donnees = np.zeros(graphes.dimensions(env), dtype=int)
                donnees[0, 0] = 2
                donnees[1, 1] = 20
                fig, ax = plt.subplots()
                try:
                    image = graphes.afficher_comptages(ax, donnees, env, 20)
                    self.assertIsInstance(image.norm, LogNorm)
                    self.assertEqual(image.norm.vmin, 1)
                    self.assertEqual(image.norm.vmax, 20)
                    self.assertAlmostEqual(float(image.norm(1)), 0.0)
                    self.assertAlmostEqual(float(image.norm(20)), 1.0)
                    self.assertTrue(np.ma.getmaskarray(image.get_array())[-1, -1])
                finally:
                    plt.close(fig)

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
        config = dict(experience.EXPERIENCES_BRAS[0], nb_episodes=100, max_pas=12)
        par_script, _ = experience.entrainer(config, 42)

        np.random.seed(42)
        env = MODES_BRAS["naive"](depart_aleatoire=True, seed=42)
        complet = QLearning(environnement=env, alpha=config["alpha"], gamma=config["gamma"],
                            epsilon=1.0, epsilon_min=config["epsilon_min"],
                            decroissance=config["decroissance"], nb_episodes=100, max_pas=12)
        complet.apprendre()

        self.assertTrue(np.array_equal(par_script.Q, complet.Q))


    def test_entrainement_fractionne_et_suivi_identiques_au_complet(self):
        """Évaluer à mi-parcours ne doit modifier ni Q, ni epsilon, ni l'aléatoire."""
        config = dict(experience.EXPERIENCES_BRAS[0], nb_episodes=100, max_pas=12)

        temoin, env_temoin = experience.entrainer(config, 42)
        etat_aleatoire_temoin = np.random.get_state()
        rng_environnement_temoin = env_temoin._rng.getstate()

        suivi, env_suivi, bilans = experience.entrainer_avec_suivi(
            config, 42, pas_evaluation=25, nb_tests=5
        )
        self.assertEqual([b["episode"] for b in bilans], [25, 50, 75, 100])
        self.assertTrue(all(b["tests"] == 5 for b in bilans))
        self.assertTrue(all(b["reussites"] + b["echecs"] == 5 for b in bilans))
        self.assertTrue(all(b["max_pas"] == 12 for b in bilans))
        self.assertTrue(all(0 <= b["taux_reussite"] <= 100 for b in bilans))

        np.testing.assert_array_equal(temoin.Q, suivi.Q)
        np.testing.assert_array_equal(temoin.visites, suivi.visites)
        np.testing.assert_array_equal(temoin.mises_a_jour, suivi.mises_a_jour)
        np.testing.assert_array_equal(temoin.explorations, suivi.explorations)
        np.testing.assert_array_equal(temoin.somme_epsilons, suivi.somme_epsilons)
        self.assertEqual(temoin.epsilon, suivi.epsilon)
        self.assertEqual(temoin.nb_episodes, suivi.nb_episodes)
        self.assertEqual(temoin.recompenses, suivi.recompenses)
        self.assertEqual(temoin.nombre_actions, suivi.nombre_actions)
        self.assertEqual(temoin.epsilons, suivi.epsilons)
        self.assertEqual(temoin.taux_exploration, suivi.taux_exploration)
        self.assertEqual(env_temoin.etat, env_suivi.etat)
        self.assertEqual(rng_environnement_temoin, env_suivi._rng.getstate())
        etat_aleatoire_suivi = np.random.get_state()
        self.assertEqual(etat_aleatoire_temoin[0], etat_aleatoire_suivi[0])
        np.testing.assert_array_equal(etat_aleatoire_temoin[1], etat_aleatoire_suivi[1])
        self.assertEqual(etat_aleatoire_temoin[2:], etat_aleatoire_suivi[2:])

    def test_evaluation_sans_modification_agent(self):
        agent = agent_bras(nb_episodes=5)
        agent.apprendre()
        avant = agent.Q.copy()
        compteurs = agent.visites.copy()
        etat = agent.env.etat
        np_alea = np.random.get_state()
        rng_env = agent.env._rng.getstate()

        bilan = experience.evaluer_politique(agent, agent.env.etats_depart[:4])
        self.assertEqual(bilan["tests"], 4)
        self.assertEqual(bilan["reussites"] + bilan["echecs"], 4)
        np.testing.assert_array_equal(agent.Q, avant)
        np.testing.assert_array_equal(agent.visites, compteurs)
        self.assertEqual(agent.env.etat, etat)
        self.assertEqual(agent.env._rng.getstate(), rng_env)
        etat_apres = np.random.get_state()
        np.testing.assert_array_equal(np_alea[1], etat_apres[1])
        self.assertEqual(np_alea[2:], etat_apres[2:])

    def test_graphe_reussite_avec_et_sans_reussites(self):
        """La progression se trace même quand un mode ne réussit jamais."""
        def bilan(episode, pas, tests=5):
            return dict(episode=episode, tests=tests, max_pas=12,
                        reussites=len(pas), echecs=tests - len(pas),
                        taux_reussite=100 * len(pas) / tests,
                        mouvements_reussis=pas)

        suivis = {
            "naive": [bilan(50, []), bilan(100, [])],
            "smart": [bilan(50, [8]), bilan(100, [4, 6, 7])],
            "distance": [bilan(50, [11]), bilan(100, [4, 9])],
        }
        with mock.patch.object(Figure, "savefig") as enregistrer:
            graphes.graphe_progression_reussite(suivis, "progression.png")
        self.assertEqual(enregistrer.call_count, 1)

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

    def test_meilleure_action_couleurs_bras_et_fleches_labyrinthe(self):
        """Bras : quatre couleurs catégorielles, sans textes ni chemin.
        Labyrinthe : flèches et chemin conservés.
        """
        agent = agent_bras(nb_episodes=5)
        agent.apprendre()
        with (mock.patch.object(Axes, "imshow", autospec=True) as images,
              mock.patch.object(Axes, "text", autospec=True) as textes,
              mock.patch.object(Axes, "arrow", autospec=True) as fleches,
              mock.patch.object(Axes, "plot", autospec=True) as chemins,
              mock.patch.object(Axes, "add_patch", autospec=True) as patches,
              mock.patch.object(Figure, "legend", autospec=True) as legendes,
              mock.patch.object(Figure, "savefig"),
              mock.patch.object(graphes, "chemin_glouton",
                                side_effect=AssertionError("Pas de chemin pour le bras"))):
            graphes.graphe_meilleure_action(agent, agent.env, "Bras", "inutilise.png")

            self.assertEqual(images.call_count, 1)
            carte = np.asarray(images.call_args.args[1])
            self.assertEqual(carte.shape, (36, 36))
            self.assertTrue(np.isin(carte[np.isfinite(carte)], [0, 1, 2, 3]).all())

            couleurs = images.call_args.kwargs["cmap"]
            self.assertEqual(couleurs.N, 4)
            self.assertEqual(len(couleurs.colors), 4)
            self.assertEqual(list(images.call_args.kwargs["norm"].boundaries),
                             [-0.5, 0.5, 1.5, 2.5, 3.5])
            self.assertTrue(np.allclose(couleurs.get_bad(),
                                        to_rgba(graphes.SANS_DONNEE)))

            for arrivee in agent.env.etats_objectif:
                self.assertTrue(np.isnan(carte.flat[arrivee]))
            for etat in range(agent.env.nb_etats):
                if agent.visites[etat] > 0 and agent.env.actions_possibles(etat):
                    self.assertEqual(carte.flat[etat], agent.meilleure_action(etat))
                else:
                    self.assertTrue(np.isnan(carte.flat[etat]))

            textes.assert_not_called()
            fleches.assert_not_called()
            chemins.assert_not_called()
            self.assertEqual(patches.call_count, 2 * len(agent.env.etats_objectif))
            self.assertTrue(all(isinstance(c.args[1], graphes.Circle)
                                for c in patches.call_args_list))
            self.assertEqual(legendes.call_count, 1)
            noms_legende = [h.get_label()
                           for h in legendes.call_args.kwargs["handles"]]
            self.assertEqual(noms_legende[:4],
                             ["theta1+", "theta1-", "theta2+", "theta2-"])
            self.assertIn("arrivée", noms_legende)

        np.random.seed(0)
        env = Labyrinthe5x5()
        agent_labyrinthe = QLearning(environnement=env, alpha=0.1, gamma=0.9,
                                    epsilon=1.0, epsilon_min=0.05, decroissance=0.99,
                                    nb_episodes=10, max_pas=15)
        agent_labyrinthe.apprendre()
        with (mock.patch.object(Axes, "text", autospec=True) as textes,
              mock.patch.object(Axes, "arrow", autospec=True) as fleches,
              mock.patch.object(Axes, "plot", autospec=True) as chemins,
              mock.patch.object(Figure, "savefig")):
            graphes.graphe_meilleure_action(agent_labyrinthe, env,
                                            "Labyrinthe", "inutilise.png")
            self.assertGreater(fleches.call_count, 0)
            self.assertGreater(chemins.call_count, 0)
            self.assertFalse(any("theta" in str(appel)
                                 for appel in textes.call_args_list))


class TestNeufActions(unittest.TestCase):
    def agent_9(self, nb_episodes=5):
        np.random.seed(42)
        env = BrasRobotNaive(depart_aleatoire=True, seed=42, nb_actions=9)
        agent = QLearning(environnement=env, alpha=0.2, gamma=0.95, epsilon=1.0,
                          epsilon_min=0.15, decroissance=0.9995,
                          nb_episodes=nb_episodes, max_pas=12)
        agent.apprendre()
        return agent

    def test_cartes_par_action_3x3(self):
        """Neuf cartes Q et neuf cartes de passages, une par action."""
        agent = self.agent_9()
        self.assertEqual(graphes.disposition_actions(4), (2, 2))
        self.assertEqual(graphes.disposition_actions(9), (3, 3))
        for fonction in (graphes.graphe_cartes_q_brutes, graphes.graphe_passages_par_action):
            with self.subTest(fonction=fonction.__name__):
                with (mock.patch.object(Axes, "imshow", autospec=True,
                                        wraps=Axes.imshow) as images,
                      mock.patch.object(Figure, "savefig")):
                    fonction(agent, agent.env, "Bras", "inutilise.png")
                self.assertEqual(images.call_count, 9)

    def test_meilleure_action_decomposee_par_articulation(self):
        """Deux cartes : variation de theta1 et de theta2 dans {-1, 0, +1}."""
        agent = self.agent_9()
        with (mock.patch.object(Axes, "imshow", autospec=True) as images,
              mock.patch.object(Figure, "savefig")):
            graphes.graphe_meilleure_action(agent, agent.env, "Bras", "inutilise.png")
        self.assertEqual(images.call_count, 2)
        cartes = [np.asarray(appel.args[1]) for appel in images.call_args_list]
        for etat in range(agent.env.nb_etats):
            ligne, colonne = divmod(etat, agent.env.nb_theta2)
            if agent.visites[etat] > 0 and agent.env.actions_possibles(etat):
                attendu = agent.env.deplacements[agent.meilleure_action(etat)]
                self.assertEqual((cartes[0][ligne, colonne], cartes[1][ligne, colonne]),
                                 attendu)
            else:
                self.assertTrue(np.isnan(cartes[0][ligne, colonne]))
                self.assertTrue(np.isnan(cartes[1][ligne, colonne]))

    def test_suivi_mesure_la_couverture_des_couples(self):
        config = experience.config_bras("naive", 9, nb_episodes=40, max_pas=12)
        _, env, bilans = experience.entrainer_avec_suivi(config, 42, pas_evaluation=10,
                                                         nb_tests=3)
        couvertures = [b["couples_essayes"] for b in bilans]
        self.assertEqual(couvertures, sorted(couvertures))
        self.assertTrue(all(0 < c <= 100 for c in couvertures))
        self.assertEqual(env.nb_actions, 9)


class TestEtudeEpsilon(unittest.TestCase):
    def test_mesures_et_graphes(self):
        from experiences import etude_epsilon
        resultats = etude_epsilon.etudier(modes=("naive",), epsilons=(0.0, 1.0),
                                          nb_actions=9, nb_episodes=20,
                                          pas_suivi=10, nb_tests=3)
        self.assertEqual(set(resultats["naive"]), {"constante", "aleatoire"})
        for initialisation, series in resultats["naive"].items():
            with self.subTest(initialisation=initialisation):
                self.assertEqual([b["episode"] for b in series[0.0]["evaluations"]],
                                 [10, 20])
                # epsilon constant : aucune action au hasard, ou uniquement au hasard.
                self.assertEqual(series[0.0]["final"]["actions_aleatoires"], 0.0)
                self.assertEqual(series[1.0]["final"]["actions_aleatoires"], 100.0)
        with mock.patch.object(Figure, "savefig") as enregistrer:
            graphes.graphe_epsilon_evolution(resultats, "couples_essayes", "y", "t", "a.png")
            graphes.graphe_epsilon_bilan(resultats, "b.png")
        self.assertEqual(enregistrer.call_count, 2)

    def test_epsilon_constant_dans_la_configuration(self):
        from experiences import etude_epsilon
        config = etude_epsilon.config_epsilon("smart", 0.2, "constante", nb_actions=9)
        agent, _ = experience.creer_agent(config, 42)
        self.assertEqual((agent.epsilon, agent.epsilon_min, agent.decroissance),
                         (0.2, 0.2, 1.0))
        self.assertTrue((agent.Q == 0).all())

    def test_moyenne_seeds(self):
        from experiences import etude_epsilon
        a = {"evaluations": [{"episode": 10, "taux_reussite": 0.0, "couples_essayes": 10.0}],
             "final": {"taux_reussite": 0.0}}
        b = {"evaluations": [{"episode": 10, "taux_reussite": 50.0, "couples_essayes": 30.0}],
             "final": {"taux_reussite": 100.0}}
        moyenne = etude_epsilon.moyenne_seeds([a, b])
        self.assertEqual(moyenne["evaluations"][0]["couples_essayes"], 20.0)
        self.assertEqual(moyenne["final"]["taux_reussite"], 50.0)


if __name__ == "__main__":
    unittest.main()
