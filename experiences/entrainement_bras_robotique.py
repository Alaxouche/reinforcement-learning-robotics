"""Entrainement Q-learning : bras robotique plan à deux articulations.

Lancement : py -m experiences.entrainement_bras_robotique
"""
import argparse

import numpy as np

from src.algorithmes.q_learning import QLearning
from src.environnements.bras_robotique.bras_robotique_2d import BrasRobotique2D


def suivre_politique(algorithme):
    """Suit la politique gloutonne sans exploration ni apprentissage."""
    env = algorithme.env
    etat = env.reset()
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
        mode_recompense=mode, coefficient_distance=coefficient_distance
    )
    algorithme = QLearning(
        environnement=environnement,
        alpha=0.2, gamma=0.95, epsilon=1.0,
        epsilon_min=0.05, decroissance=0.996,
        nb_episodes=5000, max_pas=50
    )
    algorithme.apprendre()
    etats, reussi = suivre_politique(algorithme)

    print(f"\nBRAS ROBOTIQUE - RÉCOMPENSE {mode.upper()}")
    print(f"Seed : {seed}")
    if mode == "smart":
        print(f"Coefficient de distance : {coefficient_distance}")
    print(f"Cible M0 : {environnement.cible}")
    print(f"États discrets : {environnement.nb_etats}")
    print(f"Cible atteinte : {'oui' if reussi else 'non'}")
    print(f"Nombre de mouvements : {len(etats) - 1}")
    print(f"Angles finaux (degrés) : {environnement.angles(etats[-1])}")
    print(f"Position finale : {environnement.positions(etats[-1])[2]}")
    print(f"Distance à la cible : {environnement.distance_cible(etats[-1]):.4f}")
    return algorithme, reussi, len(etats) - 1


def main(mode="tous", coefficient_distance=2.0, seed=42):
    modes = ("naive", "smart") if mode == "tous" else (mode,)
    resultats = {}
    for nom in modes:
        resultats[nom] = entrainer_mode(
            nom, coefficient_distance=coefficient_distance, seed=seed)
    if len(resultats) == 2:
        print("\nCOMPARAISON : MÊME BRAS, MÊMES PARAMÈTRES")
        for nom in modes:
            _, reussi, actions = resultats[nom]
            print(f"{nom:>5}: cible atteinte = {reussi}, actions = {actions}")
        print("Attention : les récompenses ne sont pas directement comparables.")
    return resultats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comparer les récompenses naive et smart du bras robotique.")
    parser.add_argument("--mode", choices=("naive", "smart", "tous"), default="tous")
    parser.add_argument("--coefficient", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    main(mode=arguments.mode, coefficient_distance=arguments.coefficient,
         seed=arguments.seed)
