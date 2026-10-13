"""Génère 5 graphes d'analyse pour chaque labyrinthe.

    1_convergence.png        récompense, pas et epsilon par épisode
    2_epsilon_seeds.png      impact d'epsilon sur la répétabilité (seeds différentes)
    3_surfaces_q.png         Q(s, a) en 3D pour chaque action
    4_meilleure_action.png   meilleure action dans chaque case et chemin suivi
    5_carte_valeurs.png      carte couleur de V(s) normalisée entre 0 et 1

Les images sont écrites dans resultats/<labyrinthe>/.
Lancement depuis la racine du projet : py -m experiences.graphes
"""

from pathlib import Path

import numpy as np

from src.algorithmes.q_learning import QLearning
from src.environnements.labyrinthe_3x3 import Labyrinthe3x3
from src.environnements.labyrinthe_5x5 import Labyrinthe5x5
from src.environnements.labyrinthe_9x9 import Labyrinthe9x9
from src.visualisation import graphes


DOSSIER_RESULTATS = Path(__file__).resolve().parent.parent / "resultats"

SEEDS = range(10)  # 10 entraînements différents par stratégie

# Réglages adaptés à la taille de chaque labyrinthe.
# "bloc" = nombre d'épisodes entre deux vérifications du chemin appris.
LABYRINTHES = [
    {"nom": "Labyrinthe 3x3", "dossier": "labyrinthe_3x3", "classe": Labyrinthe3x3,
     "nb_episodes": 300, "max_pas": 20, "bloc": 5, "decroissance": 0.98},
    {"nom": "Labyrinthe 5x5", "dossier": "labyrinthe_5x5", "classe": Labyrinthe5x5,
     "nb_episodes": 1000, "max_pas": 30, "bloc": 10, "decroissance": 0.99},
    {"nom": "Labyrinthe 9x9", "dossier": "labyrinthe_9x9", "classe": Labyrinthe9x9,
     "nb_episodes": 3000, "max_pas": 200, "bloc": 25, "decroissance": 0.998},
]

# Stratégies d'exploration comparées sur le graphe 2 : (label, epsilon de départ, epsilon min).
# Quand epsilon de départ = epsilon min, epsilon reste fixe.
STRATEGIES = [
    ("epsilon = 0.05 fixe", 0.05, 0.05),
    ("epsilon = 0.2 fixe", 0.2, 0.2),
    ("epsilon = 0.5 fixe", 0.5, 0.5),
    ("epsilon 1 -> 0.05", 1.0, 0.05),
]


def entrainer(config, epsilon, epsilon_min, seed):
    """Entraîne un agent neuf et mesure à quel épisode il a convergé.

    L'entraînement est découpé en blocs de config["bloc"] épisodes.
    Après chaque bloc, on suit le chemin appris et on note son retour.
    L'épisode de convergence est la fin du premier bloc à partir duquel
    ce retour ne change plus jusqu'à la fin.
    """
    environnement = config["classe"]()
    algorithme = QLearning(
        environnement=environnement,
        alpha=0.1,
        gamma=0.9,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        decroissance=config["decroissance"],
        nb_episodes=config["bloc"],
        max_pas=config["max_pas"],
        seed=seed,
    )

    # apprendre() vide ses listes à chaque appel : on les cumule ici.
    recompenses, longueurs, epsilons, retours = [], [], [], []

    nb_blocs = config["nb_episodes"] // config["bloc"]
    for _ in range(nb_blocs):
        algorithme.apprendre()  # reprend avec la table Q et l'epsilon actuels
        recompenses += algorithme.recompenses
        longueurs += algorithme.longueurs
        epsilons += algorithme.epsilons
        _, retour, _ = graphes.chemin_glouton(algorithme, environnement)
        retours.append(retour)

    algorithme.recompenses = recompenses
    algorithme.longueurs = longueurs
    algorithme.epsilons = epsilons

    # Premier bloc à partir duquel tous les retours sont égaux au dernier.
    convergence = config["nb_episodes"]
    for i in range(nb_blocs):
        if all(r == retours[-1] for r in retours[i:]):
            convergence = (i + 1) * config["bloc"]
            break

    return algorithme, environnement, convergence


def comparer_epsilons(config):
    """Entraîne chaque stratégie sur chaque seed (données du graphe 2)."""
    resultats = []

    for label, epsilon, epsilon_min in STRATEGIES:
        recompenses, convergences, retours = [], [], []

        for seed in SEEDS:
            algorithme, environnement, convergence = entrainer(config, epsilon, epsilon_min, seed)
            recompenses.append(algorithme.recompenses)
            convergences.append(convergence)
            retours.append(graphes.chemin_glouton(algorithme, environnement)[1])

        resultats.append({"label": label, "recompenses": recompenses,
                          "convergences": convergences})

        print(f"    {label:<22} convergence : moyenne {np.mean(convergences):7.1f}, "
              f"écart-type {np.std(convergences):6.1f} | retour glouton final : "
              f"{np.mean(retours):.2f} +/- {np.std(retours):.2f}")

    return resultats


def main():
    for config in LABYRINTHES:
        nom = config["nom"]
        dossier = DOSSIER_RESULTATS / config["dossier"]
        dossier.mkdir(parents=True, exist_ok=True)
        print(f"\n{nom}")

        # Agent de référence pour les graphes 1, 3, 4 et 5 : epsilon 1 -> 0.05, seed 0.
        algorithme, environnement, _ = entrainer(config, 1.0, 0.05, seed=0)

        graphes.graphe_convergence(algorithme, nom, dossier / "1_convergence.png")
        graphes.graphe_epsilon_seeds(comparer_epsilons(config), nom,
                                     dossier / "2_epsilon_seeds.png")
        graphes.graphe_surfaces_q(algorithme, environnement, nom, dossier / "3_surfaces_q.png")
        graphes.graphe_meilleure_action(algorithme, environnement, nom,
                                        dossier / "4_meilleure_action.png")
        graphes.graphe_carte_valeurs(algorithme, environnement, nom,
                                     dossier / "5_carte_valeurs.png")

        print(f"    graphes écrits dans {dossier}")


if __name__ == "__main__":
    main()
