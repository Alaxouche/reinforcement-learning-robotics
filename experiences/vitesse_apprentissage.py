"""Deux graphiques pour comparer la vitesse d'apprentissage des récompenses.

Depuis la racine du dépôt :
    py -m experiences.vitesse_apprentissage
    py -m experiences.vitesse_apprentissage --actions 9

Q(s,a) commence à zéro dans chaque entraînement indépendant.
Mêmes hyperparamètres et mêmes départs de test pour naive, smart, distance.
La réussite est évaluée sans exploration ; les mouvements comptés pendant
l'entraînement incluent l'exploration epsilon-greedy.
"""

import argparse
from pathlib import Path

from experiences.entrainement_bras_robotique import MODES_BRAS
from experiences.graphes import DOSSIER_RESULTATS, config_bras, entrainer_avec_suivi
from src.visualisation import graphes


def generer(nb_actions=9, nb_episodes=5000, seed=42, pas_evaluation=500,
            nb_tests=30, fenetre=100, dossier=None):
    """Entraîne les trois récompenses et trace deux courbes comparatives.

    Renvoie les chemins des deux images, les évaluations et les agents.
    Le taux de réussite correspond à une politique gloutonne évaluée sans
    exploration, sur les mêmes départs, avec un horizon de 50 mouvements.
    Les mouvements par épisode sont mesurés PENDANT l'apprentissage :
    un épisode court ne prouve pas, à lui seul, un meilleur comportement.
    """
    if nb_actions not in (4, 9):
        raise ValueError("nb_actions doit valoir 4 ou 9.")
    if min(nb_episodes, pas_evaluation, nb_tests, fenetre) <= 0:
        raise ValueError("Episodes, pas d'évaluation, tests et fenêtre doivent être positifs.")

    suivis = {}
    algorithmes = {}
    for mode in MODES_BRAS:
        config = config_bras(
            mode, nb_actions=nb_actions, depart_aleatoire=True,
            nb_episodes=nb_episodes
        )
        agent, _, evaluations = entrainer_avec_suivi(
            config, seed=seed, pas_evaluation=pas_evaluation, nb_tests=nb_tests
        )
        suivis[mode] = evaluations
        algorithmes[mode] = agent

    destination = (Path(dossier) if dossier is not None
                   else DOSSIER_RESULTATS / "bras_robotique" / f"{nb_actions}_actions")
    destination.mkdir(parents=True, exist_ok=True)
    fichier_reussite = destination / "comparaison_reussite_apprentissage.png"
    fichier_mouvements = destination / "comparaison_longueur_episodes.png"

    graphes.graphe_progression_reussite(
        suivis, fichier_reussite,
        titre=f"Bras à {nb_actions} actions : taux de réussite selon la récompense"
    )
    graphes.graphe_longueur_episodes_bras(
        algorithmes, fichier_mouvements, fenetre=fenetre,
        titre=f"Bras à {nb_actions} actions : mouvements par épisode selon la récompense"
    )
    return {
        "reussite": fichier_reussite,
        "mouvements": fichier_mouvements,
        "suivis": suivis,
        "algorithmes": algorithmes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Vitesse d'apprentissage du bras : naive, smart, distance."
    )
    parser.add_argument("--actions", choices=("4", "9", "tous"), default="tous")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pas-evaluation", type=int, default=500)
    parser.add_argument("--tests", type=int, default=30)
    parser.add_argument("--fenetre", type=int, default=100)
    args = parser.parse_args()

    nombres_actions = (4, 9) if args.actions == "tous" else (int(args.actions),)
    for actions in nombres_actions:
        generer(actions, nb_episodes=args.episodes, seed=args.seed,
                pas_evaluation=args.pas_evaluation, nb_tests=args.tests,
                fenetre=args.fenetre)
        print(f"Comparaison des récompenses ({actions} actions) : graphiques enregistrés.")


if __name__ == "__main__":
    main()
