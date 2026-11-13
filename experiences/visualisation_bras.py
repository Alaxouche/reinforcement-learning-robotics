"""Fenêtre avec le bras animé, pour le modèle choisi.

    py -m experiences.visualisation_bras distance
    py -m experiences.visualisation_bras smart
    py -m experiences.visualisation_bras naive
"""

import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from experiences.animation_bras import creer_agent, points_bras, trajectoire_q
from experiences.entrainement_bras_robotique import MODES_BRAS


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "distance"
    if mode not in MODES_BRAS:
        sys.exit(f"Modèle inconnu : {mode} (choix : {', '.join(MODES_BRAS)})")

    print(f"Entraînement du modèle {mode}...")
    agent = creer_agent(mode)
    env = agent.env
    etats, actions, _, _ = trajectoire_q(agent, env.etat_initial)

    # Angles de chaque position, avec quelques images entre deux positions.
    poses = [env.angles(etats[0])]
    for action in actions:
        di, dj = env.deplacements[action]
        t1, t2 = poses[-1]
        poses.append((t1 + di * env.pas_deg, t2 + dj * env.pas_deg))
    images = []
    for (a1, a2), (b1, b2) in zip(poses, poses[1:]):
        for k in range(6):
            t = k / 6
            images.append((a1 + t * (b1 - a1), a2 + t * (b2 - a2)))
    images += [poses[-1]] * 20

    fig, ax = plt.subplots(figsize=(7, 7))
    taille = env.l1 + env.l2 + 0.3
    ax.set_xlim(-taille, taille)
    ax.set_ylim(-taille, taille)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(f"Bras robotique — modèle {mode}")
    ax.add_patch(Circle(env.cible, env.tolerance, color="red", alpha=0.2))
    ax.plot(*env.cible, "rx", markersize=10)
    bras, = ax.plot([], [], "o-", linewidth=6, markersize=10, color="tab:blue")
    trace, = ax.plot([], [], ":", color="gray")
    xs, ys = [], []

    def dessiner(i):
        if i == 0:
            xs.clear()
            ys.clear()
        o, coude, main_ = points_bras((env.l1, env.l2), images[i])
        bras.set_data([o[0], coude[0], main_[0]], [o[1], coude[1], main_[1]])
        xs.append(main_[0])
        ys.append(main_[1])
        trace.set_data(xs, ys)
        return bras, trace

    animation = FuncAnimation(fig, dessiner, frames=len(images), interval=40)
    plt.show()


if __name__ == "__main__":
    main()
