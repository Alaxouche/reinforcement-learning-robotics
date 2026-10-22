"""Entrainement Q-learning : bras robotique plan à deux articulations.

Lancement : py -m experiences.entrainement_bras_robotique
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.algorithmes.q_learning import QLearning
from src.environnements.bras_robotique.bras_robotique_2d import BrasRobotique2D
from experiences.graphiques import generer_graphiques
from experiences.graphes_analyse import generer_graphes_analyse


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


def dessiner_trajectoire(env, etats, destination):
    """Trajectoire dans le plan physique de l'effecteur (x,y)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    p0, pf = env.positions(etats[0]), env.positions(etats[-1])
    ax.plot([p[0] for p in p0], [p[1] for p in p0], "--",
            color="gray", lw=3, label="Bras initial")
    ax.plot([p[0] for p in pf], [p[1] for p in pf], "-o",
            color="tab:blue", lw=4, label="Bras final")
    trajectoire = [env.positions(etat)[2] for etat in etats]
    ax.plot([p[0] for p in trajectoire], [p[1] for p in trajectoire],
            "-.", color="tab:green", lw=2, label="Trajectoire de l'effecteur")
    ax.scatter(*env.cible, marker="*", s=200,
               color="tab:red", label="Cible M0", zorder=5)
    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    portee = env.l1 + env.l2
    ax.set_xlim(-portee - .2, portee + .2)
    ax.set_ylim(-portee - .2, portee + .2)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="x", ylabel="y", title="Bras robotique - trajectoire apprise")
    ax.grid(alpha=.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


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
    algorithme.apprendre(intervalle_evaluation=50)
    racine = Path(__file__).resolve().parent / "resultats" / "bras_robotique"
    if mode == "smart":
        valeur = format(coefficient_distance, ".12g").replace("-", "moins_").replace(".", "_")
        dossier = racine / mode / f"coefficient_{valeur}_seed_{seed}"
    else:
        dossier = racine / mode / f"seed_{seed}"
    images = generer_graphiques(algorithme, dossier=dossier)
    images += generer_graphes_analyse(
        algorithme, environnement, dossier, f"Bras robotique ({mode})")
    etats, reussi = suivre_politique(algorithme)
    images.append(dessiner_trajectoire(
        environnement, etats, dossier / "07_trajectoire_bras.png"))

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
    print("Graphiques :")
    for image in images:
        print(f" - {image}")
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
