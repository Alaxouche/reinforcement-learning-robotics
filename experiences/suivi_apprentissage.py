"""Évalue l'apprentissage du bras robotique tous les 500 épisodes.

Deux graphiques seulement, sans BFS ni lissage :
  1. Réussite sur TOUS les départs non terminaux.
  2. Nombre moyen de mouvements des trajets réussis.

L'évaluation suit argmax(Q) et ne modifie jamais la Q-table.
"""

import argparse

import matplotlib.pyplot as plt

from experiences.animation_bras import MODES_BRAS, creer_agent, trajectoire_q

PAS = 500


def evaluer(agent):
    """Tester la politique apprise depuis chaque pose non terminale."""
    departs = agent.env.etats_depart
    reussis = 0
    somme_mouvements = 0
    for depart in departs:
        _, actions, reussi, _ = trajectoire_q(
            agent, depart, max_pas=agent.max_pas
        )
        if reussi:
            reussis += 1
            somme_mouvements += len(actions)

    return (
        reussis,
        len(departs),
        100.0 * reussis / len(departs),
        somme_mouvements / reussis if reussis else float("nan"),
    )


def suivre(mode="distance", actions=9, episodes=5000, seed=42):
    """Même Q-learning et même epsilon dynamique que l'animation du bras."""
    if episodes <= 0:
        raise ValueError("Le nombre d'épisodes doit être positif.")

    # Le constructeur commun conserve les mêmes hyperparamètres et la même
    # initialisation Q=0. On entraîne ci-dessous, par blocs, sans repartir de zéro.
    agent = creer_agent(
        mode=mode, nb_actions=actions, episodes=episodes,
        seed=seed, entrainer=False
    )
    lignes = []

    def mesurer(n):
        reussis, total, taux, moyenne = evaluer(agent)
        lignes.append((n, taux, moyenne))
        texte_moyenne = f"{moyenne:.2f}" if moyenne == moyenne else "—"
        print(f"{n:>7} | {reussis:>4}/{total:<4} | "
              f"{taux:>7.2f} % | {texte_moyenne:>8}")

    print(f"Bras {actions} actions | récompense {mode} | "
          f"{episodes} épisodes | epsilon dynamique habituel")
    print("Épisode | Réussites   | Réussite    | Mouvements moyens (réussites)")
    print("-" * 72)

    mesurer(0)
    try:
        for debut in range(0, episodes, PAS):
            bloc = min(PAS, episodes - debut)
            agent.nb_episodes = bloc
            agent.apprendre()
            # Test déterministe : aucun tirage aléatoire, ni changement de Q.
            # L'épisode d'entraînement suivant commence par env.reset().
            mesurer(debut + bloc)
    finally:
        agent.nb_episodes = episodes
    return lignes


def tracer(lignes, mode, actions):
    """Afficher les deux courbes mesurées, sans moyenne glissante."""
    episodes = [p[0] for p in lignes]
    reussites = [p[1] for p in lignes]
    mouvements = [p[2] for p in lignes]

    fig1, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax1.plot(episodes, reussites, "-o", linewidth=2)
    ax1.set(title=f"Taux de réussite — {actions} actions — {mode}",
            xlabel="Épisodes d'entraînement", ylabel="Départs réussis (%)")
    ax1.set_xticks(episodes)
    ax1.set_ylim(0, 100)
    ax1.grid(alpha=0.3)

    fig2, ax2 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax2.plot(episodes, mouvements, "-o", linewidth=2)
    ax2.set(title=f"Mouvements des trajets réussis — {actions} actions — {mode}",
            xlabel="Épisodes d'entraînement",
            ylabel="Nombre moyen de mouvements (réussites uniquement)")
    ax2.set_xticks(episodes)
    ax2.grid(alpha=0.3)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Tracer la réussite et les mouvements tous les 500 épisodes."
    )
    parser.add_argument("--mode", choices=tuple(MODES_BRAS), default="distance")
    parser.add_argument("--actions", type=int, choices=(4, 9), default=9)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes doit être positif.")
    lignes = suivre(args.mode, args.actions, args.episodes, args.seed)
    tracer(lignes, args.mode, args.actions)


if __name__ == "__main__":
    main()
