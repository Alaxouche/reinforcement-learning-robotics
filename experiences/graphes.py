"""Génère les graphes d'analyse du labyrinthe et du bras robotique.

    3_cartes_q_brutes.png        Q(s, a) brute en 2D, une carte par action
    4_meilleure_action.png       meilleure action dans chaque état et chemin suivi
    6_visites.png                nombre de passages par état
    comparaison_reussite_apprentissage.png  taux de réussite, tous les 500 épisodes
    comparaison_mouvements_finaux.png       mouvements réussis et échecs après 5 000 épisodes

Les cartes sont écrites dans experiences/resultats/labyrinthes/ et
experiences/resultats/bras_robotique/<mode>/ (naive, smart, distance).
Les deux comparaisons des récompenses se trouvent dans
experiences/resultats/bras_robotique/.
Lancement depuis la racine du projet : py -m experiences.graphes
"""

from pathlib import Path
import random

import numpy as np

from experiences.entrainement_bras_robotique import MODES_BRAS, suivre_politique
from src.algorithmes.q_learning import QLearning
from src.environnements.labyrinthe.labyrinthe_5x5 import Labyrinthe5x5
from src.visualisation import graphes


DOSSIER_RESULTATS = Path(__file__).resolve().parent / "resultats"

SEED_REFERENCE = 42  # même seed que experiences.entrainement
PAS_EVALUATION = 500
NB_DEPARTS_TEST = 30

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

def creer_agent(config, seed):
    """Construit une expérience déterministe, sans commencer l'entraînement."""
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
    return algorithme, environnement


def entrainer(config, seed):
    """Entraîne une fois avec les mêmes paramètres que le script de référence."""
    algorithme, environnement = creer_agent(config, seed)
    algorithme.apprendre()
    return algorithme, environnement


def evaluer_politique(algorithme, departs_test):
    """Évalue la politique gloutonne sans changer Q, les compteurs ou le hasard.

    Tous les modes reçoivent les mêmes états non terminaux ; une réussite
    nécessite d'atteindre la cible en au plus max_pas mouvements.
    """
    if not departs_test:
        raise ValueError("Il faut au moins un état de test.")
    environnement = algorithme.env
    etat_avant = environnement.etat
    mouvements_reussis = []
    try:
        for depart in departs_test:
            etats, reussi = suivre_politique(algorithme, depart)
            if reussi:
                mouvements_reussis.append(len(etats) - 1)
    finally:
        environnement.etat = etat_avant

    reussites = len(mouvements_reussis)
    return {
        "tests": len(departs_test),
        "reussites": reussites,
        "echecs": len(departs_test) - reussites,
        "taux_reussite": 100.0 * reussites / len(departs_test),
        "mouvements_reussis": mouvements_reussis,
    }


def entrainer_avec_suivi(config, seed, pas_evaluation=PAS_EVALUATION,
                         nb_tests=NB_DEPARTS_TEST):
    """Entraîne une seule fois, en évaluant à intervalles réguliers.

    Les Q-values, epsilon, générateurs aléatoires et compteurs sont conservés
    entre les blocs d'épisodes. L'évaluation ne consomme aucun tirage aléatoire.
    """
    if pas_evaluation <= 0 or nb_tests <= 0:
        raise ValueError("Le pas d'évaluation et le nombre de tests doivent être positifs.")

    algorithme, environnement = creer_agent(config, seed)
    if not hasattr(environnement, "etats_depart"):
        raise ValueError("Le suivi de réussite est réservé aux bras robotiques.")

    departs_test = random.Random(seed + 10_000).sample(
        environnement.etats_depart,
        k=min(nb_tests, len(environnement.etats_depart)),
    )
    total = algorithme.nb_episodes
    historique = {nom: [] for nom in (
        "recompenses", "nombre_actions", "epsilons", "taux_exploration"
    )}
    evaluations = []

    try:
        for debut in range(0, total, pas_evaluation):
            algorithme.nb_episodes = min(pas_evaluation, total - debut)
            algorithme.apprendre()
            for nom in historique:
                historique[nom].extend(getattr(algorithme, nom))

            bilan = evaluer_politique(algorithme, departs_test)
            bilan["episode"] = debut + algorithme.nb_episodes
            evaluations.append(bilan)
    finally:
        algorithme.nb_episodes = total

    for nom, valeurs in historique.items():
        setattr(algorithme, nom, valeurs)
    return algorithme, environnement, evaluations


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
    suivis_bras = {}
    # Le graphique de comparaison labyrinthe / bras est retiré.
    (DOSSIER_RESULTATS / "comparaison_labyrinthe_bras.png").unlink(missing_ok=True)

    for config in EXPERIENCES:
        nom = config["nom"]
        dossier = DOSSIER_RESULTATS / config["dossier"]
        dossier.mkdir(parents=True, exist_ok=True)
        # Évite de conserver des images obsolètes créées par l'ancienne version.
        for ancien in ("1_convergence.png", "2_epsilon_seeds.png", "5_cartes_q.png",
                       "5b_carte_valeurs.png", "7_carte_epsilon.png",
                       "8_mises_a_jour.png", "9_lien_epsilon_greedy.png"):
            (dossier / ancien).unlink(missing_ok=True)
        print(f"\n{nom}")

        # Chaque bras est entraîné une seule fois ; les évaluations intermédiaires
        # sont déterministes et ne changent pas l'entraînement ni la Q-table finale.
        if config["dossier"].parts[0] == "bras_robotique":
            algorithme, environnement, evaluations = entrainer_avec_suivi(
                config, SEED_REFERENCE
            )
            mode = config["dossier"].parts[1]
            suivis_bras[mode] = evaluations
            print(f"    succès gloutons sur {evaluations[-1]['tests']} départs : "
                  f"{evaluations[-1]['reussites']}/{evaluations[-1]['tests']}")
        else:
            algorithme, environnement = entrainer(config, SEED_REFERENCE)
        agents.append((nom, algorithme, environnement))

        graphes.graphe_cartes_q_brutes(algorithme, environnement, nom,
                                       dossier / "3_cartes_q_brutes.png")
        graphes.graphe_meilleure_action(algorithme, environnement, nom,
                                        dossier / "4_meilleure_action.png")
        graphes.graphe_visites(algorithme, environnement, nom, dossier / "6_visites.png")

        print(f"    graphes écrits dans {dossier}")

    dossier_bras = DOSSIER_RESULTATS / "bras_robotique"
    dossier_bras.mkdir(parents=True, exist_ok=True)
    graphes.graphe_progression_reussite(
        suivis_bras, dossier_bras / "comparaison_reussite_apprentissage.png"
    )
    graphes.graphe_mouvements_finaux(
        suivis_bras, dossier_bras / "comparaison_mouvements_finaux.png"
    )
    print(f"\nComparaison des récompenses : 2 graphiques dans {dossier_bras}")
    afficher_comparaison(agents)


if __name__ == "__main__":
    main()
