"""Génère les graphes d'analyse du labyrinthe et du bras robotique.

    1_convergence.png            récompense, pas et epsilon par épisode
    2_epsilon_seeds.png          impact d'epsilon sur la répétabilité (seeds différentes)
    3_cartes_q_brutes.png        Q(s, a) brute en 2D, une carte par action
    4_meilleure_action.png       meilleure action dans chaque état et chemin suivi
    5_cartes_q.png               carte couleur de Q(s, a), une par action
    5b_carte_valeurs.png         carte couleur de V(s) sur (theta1, theta2), bras seulement
    6_visites.png                nombre de passages par état
    7_carte_epsilon.png          epsilon moyen par état, en fonction de V(s)
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

SEEDS = range(10)  # 10 entraînements différents par stratégie
SEED_REFERENCE = 42  # même seed que experiences.entrainement

# Mêmes réglages que les scripts d'entraînement.
# "bloc" = nombre d'épisodes entre deux vérifications du chemin appris.
EXPERIENCES = [
    {"nom": "Labyrinthe 5x5", "dossier": Path("labyrinthes"),
     "construire": lambda seed: Labyrinthe5x5(),
     "alpha": 0.1, "gamma": 0.9, "epsilon_min": 0.05, "decroissance": 0.99,
     "nb_episodes": 1000, "max_pas": 30, "bloc": 10},
] + [
    {"nom": f"Bras robotique ({mode})", "dossier": Path("bras_robotique", mode),
     "construire": lambda seed, classe=classe: classe(depart_aleatoire=True, seed=seed),
     "alpha": 0.2, "gamma": 0.95, "epsilon_min": 0.15, "decroissance": 0.9995,
     "nb_episodes": 5000, "max_pas": 50, "bloc": 50}
    for mode, classe in MODES_BRAS.items()
]

# Stratégies d'exploration comparées sur le graphe 2 : (label, epsilon de départ, epsilon min).
# Quand epsilon de départ = epsilon min, epsilon reste fixe. La décroissance de
# l'entraînement, propre à chaque environnement, s'ajoute à cette liste.
EPSILONS_FIXES = [
    ("epsilon = 0.05 fixe", 0.05, 0.05),
    ("epsilon = 0.2 fixe", 0.2, 0.2),
    ("epsilon = 0.5 fixe", 0.5, 0.5),
]


def entrainer(config, epsilon, epsilon_min, seed):
    """Entraîne un agent neuf et mesure à quel épisode il a convergé.

    L'entraînement est découpé en blocs de config["bloc"] épisodes.
    Après chaque bloc, on suit le chemin appris et on note son retour.
    L'épisode de convergence est la fin du premier bloc à partir duquel
    ce retour ne change plus jusqu'à la fin.
    """
    np.random.seed(seed)
    environnement = config["construire"](seed)
    algorithme = QLearning(
        environnement=environnement,
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        decroissance=config["decroissance"],
        nb_episodes=config["bloc"],
        max_pas=config["max_pas"],
    )

    # apprendre() vide ses listes à chaque appel : on les cumule ici.
    recompenses, nombre_actions, epsilons, taux_exploration, retours = [], [], [], [], []

    nb_blocs = config["nb_episodes"] // config["bloc"]
    for _ in range(nb_blocs):
        algorithme.apprendre()  # reprend avec la table Q et l'epsilon actuels
        recompenses += algorithme.recompenses
        nombre_actions += algorithme.nombre_actions
        epsilons += algorithme.epsilons
        taux_exploration += algorithme.taux_exploration
        _, retour, _ = graphes.chemin_glouton(algorithme, environnement)
        retours.append(retour)

    algorithme.recompenses = recompenses
    algorithme.nombre_actions = nombre_actions
    algorithme.epsilons = epsilons
    algorithme.taux_exploration = taux_exploration

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
    decroissance = (f"epsilon 1 -> {config['epsilon_min']}", 1.0, config["epsilon_min"])

    for label, epsilon, epsilon_min in EPSILONS_FIXES + [decroissance]:
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
        print(f"\n{nom}")

        # Agent de référence pour tous les graphes sauf le 2 : l'entraînement du projet.
        algorithme, environnement, _ = entrainer(config, 1.0, config["epsilon_min"],
                                                 SEED_REFERENCE)
        agents.append((nom, algorithme, environnement))

        graphes.graphe_convergence(algorithme, nom, dossier / "1_convergence.png")
        graphes.graphe_epsilon_seeds(comparer_epsilons(config), nom,
                                     dossier / "2_epsilon_seeds.png")
        graphes.graphe_cartes_q_brutes(algorithme, environnement, nom,
                                       dossier / "3_cartes_q_brutes.png")
        graphes.graphe_meilleure_action(algorithme, environnement, nom,
                                        dossier / "4_meilleure_action.png")
        graphes.graphe_cartes_q(algorithme, environnement, nom, dossier / "5_cartes_q.png")
        if graphes.est_bras(environnement):
            graphes.graphe_carte_valeurs(algorithme, environnement, nom,
                                         dossier / "5b_carte_valeurs.png")
        graphes.graphe_visites(algorithme, environnement, nom, dossier / "6_visites.png")
        graphes.graphe_carte_epsilon(algorithme, environnement, nom,
                                     dossier / "7_carte_epsilon.png")
        graphes.graphe_mises_a_jour(algorithme, environnement, nom,
                                    dossier / "8_mises_a_jour.png")
        graphes.graphe_lien_epsilon_greedy(algorithme, environnement, nom,
                                           dossier / "9_lien_epsilon_greedy.png")

        print(f"    graphes écrits dans {dossier}")

    graphes.graphe_comparaison(agents, DOSSIER_RESULTATS / "comparaison_labyrinthe_bras.png")
    afficher_comparaison(agents)


if __name__ == "__main__":
    main()
