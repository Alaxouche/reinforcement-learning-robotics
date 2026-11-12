"""Entraînement du bras robotique : comparaison de trois récompenses.

Exécuter depuis la racine du dépôt :
  py -m experiences.entrainement_bras_robotique              # 4 et 9 actions
  py -m experiences.entrainement_bras_robotique --actions 9  # mouvements simultanés
"""
import argparse
import random

import numpy as np

from src.algorithmes.q_learning import QLearning
from src.environnements.bras_robotique.bras_robot_naive import BrasRobotNaive
from src.environnements.bras_robotique.bras_robot_smart import BrasRobotSmart
from src.environnements.bras_robotique.bras_robot_distance import BrasRobotDistance


def suivre_politique(algorithme, etat_depart):
    """Teste la politique gloutonne depuis un état précis, sans exploration."""
    env = algorithme.env
    etat = env.reset(etat_depart=etat_depart)
    etats = [etat]
    termine = env.distance_cible(etat) <= env.tolerance
    for _ in range(algorithme.max_pas):
        if termine or not env.actions_possibles(etat):
            break
        action = algorithme.meilleure_action(etat)
        etat, _, termine = env.step(action)
        etats.append(etat)
        if termine:
            break
    return etats, termine


MODES_BRAS = {
    "naive": BrasRobotNaive,
    "smart": BrasRobotSmart,
    "distance": BrasRobotDistance,
}


def entrainer_mode(mode, seed=42, nb_actions=4):
    np.random.seed(seed)
    environnement = MODES_BRAS[mode](
        longueurs=(1.0, 1.0), cible=(1.0, 1.0), pas_deg=10,
        angles_initiaux=(90, 0), tolerance=0.15,
        depart_aleatoire=True, seed=seed, nb_actions=nb_actions
    )
    algorithme = QLearning(
        environnement=environnement,
        alpha=0.2, gamma=0.95, epsilon=1.0,
        epsilon_min=0.15, decroissance=0.9995,
        nb_episodes=5000, max_pas=50
    )
    algorithme.apprendre()

    # Départ historique, conservé uniquement pour une comparaison lisible.
    etats_reference, reussi_reference = suivre_politique(
        algorithme, environnement.etat_initial
    )

    # Même liste de positions de test dans les trois modes, indépendante
    # des tirages aléatoires utilisés pendant chaque entraînement.
    rng_tests = random.Random(seed + 10_000)
    departs_test = rng_tests.sample(
        environnement.etats_depart,
        k=min(30, len(environnement.etats_depart))
    )
    succes = 0
    actions_reussies = []
    distances_finales = []
    for depart in departs_test:
        etats, reussi = suivre_politique(algorithme, depart)
        if reussi:
            succes += 1
            actions_reussies.append(len(etats) - 1)
        distances_finales.append(environnement.distance_cible(etats[-1]))

    bilan = {
        "succes": succes,
        "tests": len(departs_test),
        "taux_reussite": succes / len(departs_test),
        "moyenne_actions_reussies": (
            float(np.mean(actions_reussies)) if actions_reussies else None
        ),
        "distance_finale_moyenne": float(np.mean(distances_finales)),
    }

    print(f"\nBRAS ROBOTIQUE - RÉCOMPENSE {mode.upper()} - {nb_actions} ACTIONS")
    print(f"Seed : {seed}")
    print(f"Actions : {', '.join(environnement.noms_actions)}")
    print(f"Cible M0 : {environnement.cible}")
    print(f"États discrets : {environnement.nb_etats}")
    print("Départ aléatoire non terminal à chaque épisode d'entraînement.")
    print("Exploration : epsilon_min = 0.15, décroissance = 0.9995")
    print(f"Départ de référence (angles) : {environnement.angles(environnement.etat_initial)}")
    print(f"Cible atteinte depuis ce départ : {'oui' if reussi_reference else 'non'}")
    print(f"Nombre de mouvements (référence) : {len(etats_reference) - 1}")
    print(f"Distance finale (référence) : {environnement.distance_cible(etats_reference[-1]):.4f}")
    print(f"Réussites sur {bilan['tests']} départs de test : "
          f"{succes}/{bilan['tests']} ({100 * bilan['taux_reussite']:.1f} %)")
    if bilan["moyenne_actions_reussies"] is not None:
        print(f"Actions moyennes pour les réussites : {bilan['moyenne_actions_reussies']:.1f}")
    print(f"Distance finale moyenne : {bilan['distance_finale_moyenne']:.4f}")
    return algorithme, bilan


def main(mode="tous", seed=42, actions="tous"):
    modes = ("naive", "smart", "distance") if mode == "tous" else (mode,)
    nombres_actions = (4, 9) if actions == "tous" else (int(actions),)
    resultats = {}
    for nb_actions in nombres_actions:
        for nom in modes:
            resultats[(nom, nb_actions)] = entrainer_mode(
                nom, seed=seed, nb_actions=nb_actions
            )
    if len(resultats) > 1:
        print("\nCOMPARAISON : MÊME BRAS, MÊMES DÉPARTS DE TEST")
        for (nom, nb_actions), (_, bilan) in resultats.items():
            moyenne = bilan["moyenne_actions_reussies"]
            mouvements = f"{moyenne:.1f}" if moyenne is not None else "-"
            print(f"{nom:>8}, {nb_actions} actions : "
                  f"réussite = {100 * bilan['taux_reussite']:5.1f} %, "
                  f"mouvements (réussites) = {mouvements:>5}, "
                  f"distance finale moyenne = {bilan['distance_finale_moyenne']:.4f}")
        print("Les récompenses numériques ne sont pas directement comparables.")
        print("Un seul seed ne suffit pas pour conclure sur le meilleur mode.")
    return resultats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comparer les récompenses naive, smart et distance du bras robotique."
    )
    parser.add_argument(
        "--mode", choices=("naive", "smart", "distance", "tous"), default="tous"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--actions", choices=("4", "9", "tous"), default="tous",
        help="4 : une articulation à la fois ; 9 : les deux en même temps."
    )
    arguments = parser.parse_args()
    main(mode=arguments.mode, seed=arguments.seed, actions=arguments.actions)
