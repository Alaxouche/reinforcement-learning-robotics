"""Comparaison optionnelle d'epsilon avec plusieurs seeds.

Le graphe est distinct des entraînements normaux, car il faut réentraîner
12 agents pour 4 stratégies × 3 seeds (par défaut).

Depuis la racine :
  py -m experiences.comparer_exploration --environnement labyrinthe
  py -m experiences.comparer_exploration --environnement bras --mode smart
  py -m experiences.comparer_exploration --environnement bras --mode naive
"""

import argparse
from pathlib import Path
import numpy as np

from src.algorithmes.q_learning import QLearning
from src.environnements.labyrinthe.labyrinthe_5x5 import Labyrinthe5x5
from src.environnements.bras_robotique.bras_robotique_2d import BrasRobotique2D
from experiences.graphes_analyse import graphe_epsilon_seeds


STRATEGIES = (
    ("epsilon = 0.05 fixe", .05, .05),
    ("epsilon = 0.2 fixe", .2, .2),
    ("epsilon = 0.5 fixe", .5, .5),
    ("epsilon 1 → 0.05", 1.0, .05),
)


def comparer(environnement="labyrinthe", mode="smart", nb_episodes=None,
            nb_seeds=3, seed_initiale=42):
    """Même environnement/paramétrage, changement d'epsilon et de seed seul."""
    if nb_seeds < 1:
        raise ValueError("Le nombre de seeds doit être positif.")
    if environnement == "labyrinthe":
        nb_episodes = 1000 if nb_episodes is None else nb_episodes
        alpha, gamma, decroissance, max_pas = .1, .9, .99, 30
        dossier = Path(__file__).parent / "resultats" / "labyrinthes"
        def construire():
            return Labyrinthe5x5()
        nom = "Labyrinthe 5×5"
    elif environnement == "bras":
        nb_episodes = 1500 if nb_episodes is None else nb_episodes
        alpha, gamma, decroissance, max_pas = .2, .95, .996, 50
        dossier = Path(__file__).parent / "resultats" / "bras_robotique" / mode / "comparaison_epsilon"
        def construire():
            return BrasRobotique2D(
                longueurs=(1., 1.), cible=(1., 1.), pas_deg=10,
                angles_initiaux=(90, 0), tolerance=.15,
                mode_recompense=mode, coefficient_distance=2.)
        nom = f"Bras robotique ({mode})"
    else:
        raise ValueError("Choisir 'labyrinthe' ou 'bras'.")
    if nb_episodes <= 0:
        raise ValueError("Le nombre d'épisodes doit être positif.")
    resultat = []
    print(f"Comparaison {nom} : {len(STRATEGIES)} stratégies × {nb_seeds} seeds")
    for label, epsilon, epsilon_min in STRATEGIES:
        courbes, evaluations = [], []
        for seed in range(seed_initiale, seed_initiale + nb_seeds):
            np.random.seed(seed)
            agent = QLearning(construire(), alpha=alpha, gamma=gamma,
                              epsilon=epsilon, epsilon_min=epsilon_min,
                              decroissance=decroissance, nb_episodes=nb_episodes,
                              max_pas=max_pas)
            agent.apprendre(intervalle_evaluation=max(1, nb_episodes // 20), nb_tests=1)
            courbes.append(agent.recompenses)
            evaluations.append(agent.evaluer(nb_episodes=1))
        print(f"  {label} : score glouton final = "
              f"{np.mean(evaluations):.3f} ± {np.std(evaluations):.3f}")
        resultat.append({"label": label, "recompenses": courbes,
                         "evaluations": evaluations})
    dossier.mkdir(parents=True, exist_ok=True)
    fichier = graphe_epsilon_seeds(resultat, nom, dossier / "17_epsilon_seeds.png")
    print(f"Graphique : {fichier}")
    return fichier


def main():
    parser = argparse.ArgumentParser(description="Comparer epsilon sur plusieurs seeds.")
    parser.add_argument("--environnement", choices=("labyrinthe", "bras"), default="labyrinthe")
    parser.add_argument("--mode", choices=("naive", "smart"), default="smart")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-initiale", type=int, default=42)
    args = parser.parse_args()
    comparer(args.environnement, args.mode, args.episodes, args.seeds, args.seed_initiale)


if __name__ == "__main__":
    main()
