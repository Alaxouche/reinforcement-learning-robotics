"""Génère les graphes d'analyse du labyrinthe et du bras robotique.

    3_cartes_q_brutes.png        Q(s, a) brute en 2D, une carte par action
    4_meilleure_action.png       meilleure action dans chaque état et chemin suivi
    6_visites.png                nombre de passages par état
    8_mises_a_jour.png           nombre de mises à jour de chaque action
    9_lien_epsilon_greedy.png    lien entre ces mises à jour et epsilon-greedy

Les images sont écrites dans experiences/resultats/labyrinthes/ et
experiences/resultats/bras_robotique/<mode>/ (naive, smart, distance), et la
comparaison dans experiences/resultats/comparaison_labyrinthe_bras.png.
Lancement depuis la racine du projet : py -m experiences.graphes
"""

from pathlib import Path

import numpy as np

from experiences.entrainement_bras_robotique import MODES_BRAS
from src.algorithmes.q_learning import QLearning
from src.environnements.labyrinthe.labyrinthe_5x5 import Labyrinthe5x5
from src.visualisation import graphes


DOSSIER_RESULTATS = Path(__file__).resolve().parent / "resultats"

SEED_REFERENCE = 42  # même seed que experiences.entrainement

# Mêmes réglages que les scripts d'entraînement.
EXPERIENCES = [
    {"nom": "Labyrinthe 5x5", "dossier": Path("labyrinthes"),
     "construire": lambda seed: Labyrinthe5x5(),
     "alpha": 0.1, "gamma": 0.9, "epsilon_min": 0.05, "decroissance": 0.99,
     "nb_episodes": 1000, "max_pas": 30},
] + [
    {"nom": f"Bras robotique ({mode})", "dossier": Path("bras_robotique", mode),
     "construire": lambda seed, classe=classe: classe(depart_aleatoire=True, seed=seed),
     "alpha": 0.2, "gamma": 0.95, "epsilon_min": 0.15, "decroissance": 0.9995,
     "nb_episodes": 5000, "max_pas": 50}
    for mode, classe in MODES_BRAS.items()
]

def entrainer(config, seed):
    """Entraîne une fois avec les mêmes paramètres que le script de référence."""
    np.random.seed(seed)
    environnement = config["construire"](seed)
    algorithme = QLearning(
        environnement=environnement,
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon=1.0,
        epsilon_min=config["epsilon_min"],
        decroissance=config["decroissance"],
        nb_episodes=config["nb_episodes"],
        max_pas=config["max_pas"],
    )
    algorithme.apprendre()
    return algorithme, environnement


def afficher_comparaison(agents):
    """Affiche côte à côte les mesures de graphes.indicateurs()."""
    lignes = [
        ("états utilisables", "etats", "{:.0f}"),
        ("couples (s, a) autorisés", "couples", "{:.0f}"),
        ("mises à jour au total", "mises_a_jour", "{:.0f}"),
        ("mises à jour par couple", "mises_a_jour_par_couple", "{:.1f}"),
        ("états visités (%)", "etats_visites", "{:.1f}"),
        ("couples jamais essayés (%)", "couples_jamais_essayes", "{:.1f}"),
        ("passages sur le top 10 % des états (%)", "visites_top_10", "{:.1f}"),
        ("actions tirées au hasard (%)", "exploration", "{:.1f}"),
        ("réussite depuis tous les départs (%)", "reussite_tous_departs", "{:.1f}"),
        ("déplacements depuis le départ de référence", "deplacements", "{:.0f}"),
        ("objectif atteint depuis ce départ", "atteint", "{}"),
    ]
    mesures = [graphes.indicateurs(algorithme, environnement)
               for _, algorithme, environnement in agents]

    print("\nComparaison")
    print(f"{'':<44}" + "".join(f"{nom:>26}" for nom, _, _ in agents))
    for titre, cle, format_valeur in lignes:
        print(f"{titre:<44}" + "".join(f"{format_valeur.format(m[cle]):>26}" for m in mesures))


def main():
    agents = []

    for config in EXPERIENCES:
        nom = config["nom"]
        dossier = DOSSIER_RESULTATS / config["dossier"]
        dossier.mkdir(parents=True, exist_ok=True)
        # Évite de conserver des images obsolètes créées par l'ancienne version.
        for ancien in ("1_convergence.png", "2_epsilon_seeds.png", "5_cartes_q.png", "5b_carte_valeurs.png", "7_carte_epsilon.png"):
            (dossier / ancien).unlink(missing_ok=True)
        print(f"\n{nom}")

        # Même entraînement que les expériences de référence (seed 42).
        algorithme, environnement = entrainer(config, SEED_REFERENCE)
        agents.append((nom, algorithme, environnement))

        graphes.graphe_cartes_q_brutes(algorithme, environnement, nom,
                                       dossier / "3_cartes_q_brutes.png")
        graphes.graphe_meilleure_action(algorithme, environnement, nom,
                                        dossier / "4_meilleure_action.png")
        graphes.graphe_visites(algorithme, environnement, nom, dossier / "6_visites.png")
        graphes.graphe_mises_a_jour(algorithme, environnement, nom,
                                    dossier / "8_mises_a_jour.png")
        graphes.graphe_lien_epsilon_greedy(algorithme, environnement, nom,
                                           dossier / "9_lien_epsilon_greedy.png")

        print(f"    graphes écrits dans {dossier}")

    graphes.graphe_comparaison(agents, DOSSIER_RESULTATS / "comparaison_labyrinthe_bras.png")
    afficher_comparaison(agents)


if __name__ == "__main__":
    main()
