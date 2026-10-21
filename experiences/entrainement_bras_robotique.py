"""Entrainement Q-learning : bras robotique plan a deux articulations.

Lancement depuis la racine : py -m experiences.entrainement_bras_robotique
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.algorithmes.q_learning import QLearning
from src.environnements.bras_robotique.bras_robotique_2d import BrasRobotique2D
from experiences.graphiques import generer_graphiques


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
    """Affiche la trajectoire de l'effecteur et les poses initiale/finale."""
    fig, ax = plt.subplots(figsize=(7, 7))
    p0 = env.positions(etats[0])
    pf = env.positions(etats[-1])
    ax.plot([p[0] for p in p0], [p[1] for p in p0],
            "--", color="gray", lw=3, label="Bras initial")
    ax.plot([p[0] for p in pf], [p[1] for p in pf],
            "-o", color="tab:blue", lw=4, label="Bras final")
    trajectoire = [env.positions(etat)[2] for etat in etats]
    ax.plot([p[0] for p in trajectoire],
            [p[1] for p in trajectoire],
            "-.", color="tab:green", lw=2, label="Trajectoire de l'effecteur")
    ax.scatter(*env.cible, marker="*", s=200,
               color="tab:red", label="Cible M0", zorder=5)
    # Axes du plan, sans sol ni limite de quadrant.
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


def main():
    np.random.seed(42)

    environnement = BrasRobotique2D(
        longueurs=(1.0, 1.0),
        cible=(1.0, 1.0),
        pas_deg=10,
        angles_initiaux=(90, 0),
        tolerance=0.15
    )
    algorithme = QLearning(
        environnement=environnement,
        alpha=0.2,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        decroissance=0.996,
        nb_episodes=5000,
        max_pas=50
    )
    algorithme.apprendre(intervalle_evaluation=50)

    dossier = Path(__file__).resolve().parent / "resultats" / "bras_robotique"
    images = generer_graphiques(algorithme, dossier=dossier)
    etats, reussi = suivre_politique(algorithme)
    images.append(dessiner_trajectoire(
        environnement, etats, dossier / "07_trajectoire_bras.png"
    ))

    print("\nBRAS ROBOTIQUE 2D")
    print(f"Cible M0 : {environnement.cible}")
    print(f"Etats discrets : {environnement.nb_etats}")
    print(f"Cible atteinte : {'oui' if reussi else 'non'}")
    print(f"Nombre de mouvements : {len(etats) - 1}")
    print(f"Angles finaux (degres) : {environnement.angles(etats[-1])}")
    print(f"Position finale : {environnement.positions(etats[-1])[2]}")
    print(f"Distance a la cible : {environnement.distance_cible(etats[-1]):.4f}")
    print("\nGraphiques crees :")
    for image in images:
        print(f" - {image}")
    return algorithme


if __name__ == "__main__":
    main()
