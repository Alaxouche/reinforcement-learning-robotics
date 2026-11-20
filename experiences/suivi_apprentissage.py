"""Deux graphiques : réussite des politiques et mouvements réels par épisode.

Tous les 500 épisodes, évaluation gloutonne sur TOUS les départs possibles.
À CHAQUE épisode d'entraînement, nombre brut de mouvements avant réussite,
ou échec signalé séparément. Aucun BFS, aucune moyenne, aucun lissage.
"""

import argparse

import matplotlib.pyplot as plt

from experiences.animation_bras import MODES_BRAS, creer_agent, trajectoire_q

PAS = 500


def evaluer(agent):
    """Réussite de la politique gloutonne sur toutes les poses non terminales."""
    departs = agent.env.etats_depart
    reussites = 0
    for depart in departs:
        _, _, reussi, _ = trajectoire_q(
            agent, depart, max_pas=agent.max_pas
        )
        reussites += int(reussi)
    return reussites, len(departs), 100.0 * reussites / len(departs)


def suivre(mode="distance", actions=9, episodes=5000, seed=42):
    """Entraînement epsilon-greedy conservé, évaluations tous les 500 épisodes."""
    if episodes <= 0:
        raise ValueError("Le nombre d'épisodes doit être positif.")

    agent = creer_agent(
        mode=mode, nb_actions=actions, episodes=episodes,
        seed=seed, entrainer=False
    )
    evaluations = []
    mouvements_episodes = []  # (numéro, nombre de pas, réussite)

    def mesurer(n):
        reussis, total, taux = evaluer(agent)
        evaluations.append((n, taux))
        print(f"{n:>7} | {reussis:>4}/{total:<4} | {taux:>7.2f} %")

    print(f"Bras {actions} actions | récompense {mode} | "
          f"{episodes} épisodes | epsilon dynamique habituel")
    print("Épisode | Réussites   | Taux de réussite (évaluation sans exploration)")
    print("-" * 71)
    mesurer(0)

    try:
        for debut in range(0, episodes, PAS):
            bloc = min(PAS, episodes - debut)
            agent.nb_episodes = bloc

            def observer(mouvements, reussi):
                # Une mesure réelle par épisode, sans moyenne ni fenêtre glissante.
                mouvements_episodes.append(
                    (len(mouvements_episodes) + 1, mouvements, reussi)
                )

            agent.apprendre(suivi_episodes=observer)
            # Les évaluations n'actualisent pas Q et ne consomment pas d'aléatoire.
            mesurer(debut + bloc)
    finally:
        agent.nb_episodes = episodes

    return evaluations, mouvements_episodes, agent.max_pas


def tracer(evaluations, mouvements_episodes, max_pas, mode, actions):
    """Deux figures, mesures réelles sans lissage."""
    episodes_test = [p[0] for p in evaluations]
    taux = [p[1] for p in evaluations]

    fig1, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax1.plot(episodes_test, taux, "-o", linewidth=2)
    ax1.set(title=f"Réussite de la politique — {actions} actions — {mode}",
            xlabel="Épisodes d'entraînement",
            ylabel="Départs réussis (%) — test sans exploration")
    ax1.set_xticks(episodes_test)
    ax1.set_ylim(0, 100)
    ax1.grid(alpha=0.3)

    # Chaque point est un épisode d'entraînement et son nombre RÉEL de pas.
    x_reussis = [n for n, pas, ok in mouvements_episodes if ok]
    y_reussis = [pas for n, pas, ok in mouvements_episodes if ok]
    x_echecs = [n for n, pas, ok in mouvements_episodes if not ok]

    fig2, ax2 = plt.subplots(figsize=(10, 5), layout="constrained")
    ax2.scatter(x_reussis, y_reussis, s=9, alpha=0.6,
                label="Cible atteinte (mouvements réels)")
    if x_echecs:
        ax2.scatter(x_echecs, [max_pas + 1] * len(x_echecs),
                    marker="x", s=12, alpha=0.4, color="tab:red",
                    label="Échec (cible non atteinte)")
    ax2.set(title=f"Mouvements jusqu'à la réussite — {actions} actions — {mode}",
            xlabel="Épisode d'entraînement",
            ylabel="Mouvements effectués avant d'atteindre la cible")
    ax2.set_xlim(0, len(mouvements_episodes) + 1)
    ax2.set_ylim(0, max_pas + 3)
    ticks = list(range(0, max_pas + 1, 10))
    ax2.set_yticks(ticks + [max_pas + 1],
                   [str(t) for t in ticks] + ["Échec"])
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Réussite tous les 500 épisodes et mouvements réels par épisode."
    )
    parser.add_argument("--mode", choices=tuple(MODES_BRAS), default="distance")
    parser.add_argument("--actions", type=int, choices=(4, 9), default=9)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes doit être positif.")
    evaluations, mouvements, max_pas = suivre(
        args.mode, args.actions, args.episodes, args.seed
    )
    tracer(evaluations, mouvements, max_pas, args.mode, args.actions)


if __name__ == "__main__":
    main()
