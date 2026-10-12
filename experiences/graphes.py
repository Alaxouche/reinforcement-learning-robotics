"""Expérience : génère 5 graphes d'analyse pour chaque labyrinthe.

    1_convergence.png        récompense, pas et epsilon par épisode
    2_epsilon_seeds.png      impact d'epsilon sur la répétabilité (seeds différentes)
    3_surfaces_q.png         Q(s, a) en 3D pour chaque action
    4_meilleure_action.png   politique gloutonne argmax Q et chemin suivi
    5_carte_valeurs.png      carte couleur de V(s) normalisée entre 0 et 1

Les images sont écrites dans resultats/<labyrinthe>/.
"""

from pathlib import Path

import numpy as np

from src.algorithmes.q_learning import QLearning
from src.environnements.labyrinthe_3x3 import Labyrinthe3x3
from src.environnements.labyrinthe_5x5 import Labyrinthe5x5
from src.environnements.labyrinthe_9x9 import Labyrinthe9x9
from src.visualisation import graphes


DOSSIER_RESULTATS = Path(__file__).resolve().parent.parent / "resultats"

SEEDS = list(range(10))

# Hyperparamètres adaptés à la taille de chaque labyrinthe.
CONFIGURATIONS = [
    {
        "nom": "Labyrinthe 3x3",
        "dossier": "labyrinthe_3x3",
        "classe": Labyrinthe3x3,
        "nb_episodes": 300,
        "max_pas": 20,
        "bloc": 5,
        "decroissance": 0.98,
    },
    {
        "nom": "Labyrinthe 5x5",
        "dossier": "labyrinthe_5x5",
        "classe": Labyrinthe5x5,
        "nb_episodes": 1000,
        "max_pas": 30,
        "bloc": 10,
        "decroissance": 0.99,
    },
    {
        "nom": "Labyrinthe 9x9",
        "dossier": "labyrinthe_9x9",
        "classe": Labyrinthe9x9,
        "nb_episodes": 3000,
        "max_pas": 200,
        "bloc": 25,
        "decroissance": 0.998,
    },
]

# Stratégies d'exploration comparées : epsilon fixe ou décroissant.
STRATEGIES_EPSILON = [
    {"label": "epsilon = 0.05 fixe", "epsilon": 0.05, "epsilon_min": 0.05, "decroissant": False},
    {"label": "epsilon = 0.2 fixe", "epsilon": 0.2, "epsilon_min": 0.2, "decroissant": False},
    {"label": "epsilon = 0.5 fixe", "epsilon": 0.5, "epsilon_min": 0.5, "decroissant": False},
    {"label": "epsilon 1 -> 0.05", "epsilon": 1.0, "epsilon_min": 0.05, "decroissant": True},
]


def entrainer(config, epsilon, epsilon_min, decroissance, seed):
    """Entraîne un agent neuf sur un labyrinthe neuf, par blocs d'épisodes.

    Après chaque bloc, la politique gloutonne est évaluée. L'épisode de
    convergence est le premier à partir duquel son retour ne change plus.
    """
    environnement = config["classe"]()
    bloc = config["bloc"]

    algorithme = QLearning(
        environnement=environnement,
        alpha=0.1,
        gamma=0.9,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        decroissance=decroissance,
        nb_episodes=bloc,
        max_pas=config["max_pas"],
        seed=seed
    )

    recompenses, longueurs, epsilons, retours = [], [], [], []

    # apprendre() reprend avec la table Q et l'epsilon courants.
    for _ in range(config["nb_episodes"] // bloc):
        algorithme.apprendre()
        recompenses += algorithme.recompenses
        longueurs += algorithme.longueurs
        epsilons += algorithme.epsilons
        retours.append(graphes.chemin_glouton(algorithme, environnement)[1])

    algorithme.recompenses = recompenses
    algorithme.longueurs = longueurs
    algorithme.epsilons = epsilons
    algorithme.nb_episodes = len(recompenses)

    stable = len(retours)
    while stable > 0 and retours[stable - 1] == retours[-1]:
        stable -= 1
    convergence = (stable + 1) * bloc

    return algorithme, environnement, convergence


def comparer_epsilons(config):
    """Entraîne chaque stratégie d'epsilon sur toutes les seeds."""
    resultats = []

    for strategie in STRATEGIES_EPSILON:
        decroissance = config["decroissance"] if strategie["decroissant"] else 1.0
        recompenses = []
        convergences = []
        retours = []

        for seed in SEEDS:
            algorithme, environnement, convergence = entrainer(
                config, strategie["epsilon"], strategie["epsilon_min"], decroissance, seed
            )
            recompenses.append(algorithme.recompenses)
            convergences.append(convergence)
            retours.append(graphes.chemin_glouton(algorithme, environnement)[1])

        resultats.append({
            "label": strategie["label"],
            "recompenses": np.array(recompenses),
            "convergences": convergences,
        })

        print(f"    {strategie['label']:<22} convergence : moyenne {np.mean(convergences):7.1f}, "
              f"écart-type {np.std(convergences):6.1f} | retour glouton final : "
              f"{np.mean(retours):.2f} +/- {np.std(retours):.2f}")

    return resultats


def main():
    for config in CONFIGURATIONS:
        dossier = DOSSIER_RESULTATS / config["dossier"]
        dossier.mkdir(parents=True, exist_ok=True)
        nom = config["nom"]
        print(f"\n{nom}")

        # Agent de référence : epsilon décroissant, seed 0.
        algorithme, environnement, _ = entrainer(config, 1.0, 0.05, config["decroissance"], seed=0)

        graphes.graphe_convergence(algorithme, nom, dossier / "1_convergence.png")
        graphes.graphe_epsilon_seeds(comparer_epsilons(config), nom,
                                     dossier / "2_epsilon_seeds.png")
        graphes.graphe_surfaces_q(algorithme, environnement, nom,
                                  dossier / "3_surfaces_q.png")
        graphes.graphe_meilleure_action(algorithme, environnement, nom,
                                        dossier / "4_meilleure_action.png")
        graphes.graphe_carte_valeurs(algorithme, environnement, nom,
                                     dossier / "5_carte_valeurs.png")

        print(f"    graphes écrits dans {dossier}")


if __name__ == "__main__":
    main()
