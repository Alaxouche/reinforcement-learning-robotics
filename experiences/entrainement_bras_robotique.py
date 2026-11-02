"""Entraînement du bras robotique : comparaison de trois récompenses.

Exécuter depuis la racine du dépôt :
  py -m experiences.entrainement_bras_robotique
"""
import argparse
import random

import numpy as np

from src.algorithmes.q_learning import QLearning
from src.environnements.bras_robotique.bras_robotique_2d import BrasRobotique2D


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


def entrainer_mode(mode, coefficient_distance=2.0, seed=42):
    np.random.seed(seed)
    environnement = BrasRobotique2D(
        longueurs=(1.0, 1.0), cible=(1.0, 1.0), pas_deg=10,
        angles_initiaux=(90, 0), tolerance=0.15,
        mode_recompense=mode, coefficient_distance=coefficient_distance,
        depart_aleatoire=True, seed=seed
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

    print(f"\nBRAS ROBOTIQUE - RÉCOMPENSE {mode.upper()}")
    print(f"Seed : {seed}")
    if mode == "smart":
        print(f"Coefficient de distance : {coefficient_distance}")
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


def main(mode="tous", coefficient_distance=2.0, seed=42):
    modes = ("naive", "smart", "distance") if mode == "tous" else (mode,)
    resultats = {}
    for nom in modes:
        resultats[nom] = entrainer_mode(
            nom, coefficient_distance=coefficient_distance, seed=seed
        )
    if len(resultats) > 1:
        print("\nCOMPARAISON : MÊME BRAS, MÊMES DÉPARTS DE TEST")
        for nom in modes:
            _, bilan = resultats[nom]
            print(f"{nom:>8}: réussite = {100 * bilan['taux_reussite']:.1f} %, "
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
    parser.add_argument("--coefficient", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    main(mode=arguments.mode, coefficient_distance=arguments.coefficient,
         seed=arguments.seed)
