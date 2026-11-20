"""Suivi du Q-learning : deux mesures de la politique, sans lissage ni BFS.

- Tous les 500 épisodes : réussite depuis TOUS les départs non terminaux.
- Après CHAQUE épisode : mouvements nécessaires depuis UNE pose fixe identique,
  avec une politique gloutonne, sans epsilon et sans apprentissage lors du test.
"""

import argparse

import matplotlib.pyplot as plt

from experiences.animation_bras import (
    MODES_BRAS,
    creer_agent,
    position_depart,
    trajectoire_q,
)

PAS = 500
ANGLES_TEST = (180, 90)


def evaluer(agent):
    """Pourcentage de réussites sur tous les départs non terminaux."""
    departs = agent.env.etats_depart
    reussites = 0
    for depart in departs:
        _, _, reussi, _ = trajectoire_q(
            agent, depart, max_pas=agent.max_pas
        )
        reussites += int(reussi)
    return reussites, len(departs), 100.0 * reussites / len(departs)


def tester_depart_fixe(agent, depart):
    """Mouvements de la politique apprise, ou None si la cible n'est pas atteinte."""
    _, actions, reussi, _ = trajectoire_q(
        agent, depart, max_pas=agent.max_pas
    )
    return len(actions) if reussi else None


def suivre(mode="distance", actions=9, episodes=5000, seed=42,
           angles_test=ANGLES_TEST):
    """Entraînement à départs aléatoires, tests fixes et sans exploration."""
    if episodes <= 0:
        raise ValueError("Le nombre d'épisodes doit être positif.")

    agent = creer_agent(
        mode=mode, nb_actions=actions, episodes=episodes,
        seed=seed, entrainer=False
    )
    depart_fixe = position_depart(agent.env, angles=angles_test)
    evaluations = []
    trajets_fixes = []  # (épisode, mouvements ou None si échec)

    def tester_fixe(numero):
        # Le rejeu glouton ne modifie ni Q ni epsilon et ne tire aucun
        # aléatoire. Au prochain épisode d'entraînement, env.reset()
        # choisit à nouveau un départ aléatoire.
        trajets_fixes.append(
            (numero, tester_depart_fixe(agent, depart_fixe))
        )

    def mesurer(n):
        reussis, total, taux = evaluer(agent)
        evaluations.append((n, taux))
        valeur_fixe = trajets_fixes[-1][1]
        etat_fixe = (f"{valeur_fixe} mouvements" if valeur_fixe is not None
                     else "échec")
        print(f"{n:>7} | {reussis:>4}/{total:<4} | "
              f"{taux:>7.2f} % | {etat_fixe}")

    print(f"Bras {actions} actions | récompense {mode} | "
          f"{episodes} épisodes | epsilon dynamique habituel")
    print(f"Départ du test fixe : θ1={angles_test[0]}°, θ2={angles_test[1]}°")
    print("Épisode | Réussites   | Taux global | Test du départ fixe")
    print("-" * 70)

    # État de référence avant le moindre entraînement.
    tester_fixe(0)
    mesurer(0)

    try:
        for debut in range(0, episodes, PAS):
            agent.nb_episodes = min(PAS, episodes - debut)

            def observer(_mouvements_entrainement, _reussi_entrainement):
                # Appelé immédiatement après CHAQUE épisode de Q-learning.
                tester_fixe(len(trajets_fixes))

            agent.apprendre(suivi_episodes=observer)
            mesurer(debut + agent.nb_episodes)
    finally:
        agent.nb_episodes = episodes

    return evaluations, trajets_fixes, agent.max_pas


def tracer(evaluations, trajets_fixes, max_pas, mode, actions, angles_test):
    """Deux graphiques, les observations sont tracées sans moyenne ni lissage."""
    episodes_test = [p[0] for p in evaluations]
    taux = [p[1] for p in evaluations]

    fig1, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax1.plot(episodes_test, taux, "-o", linewidth=2)
    ax1.set(title=f"Réussite globale de la politique — {actions} actions — {mode}",
            xlabel="Épisodes d'entraînement",
            ylabel="Départs réussis (%) — sans exploration")
    ax1.set_xticks(episodes_test)
    ax1.set_ylim(0, 100)
    ax1.grid(alpha=0.3)

    # Un seul départ : les nombres de mouvements sont directement comparables.
    x = [n for n, _ in trajets_fixes]
    y = [pas if pas is not None else float("nan") for _, pas in trajets_fixes]
    echecs = [n for n, pas in trajets_fixes if pas is None]

    fig2, ax2 = plt.subplots(figsize=(10, 5), layout="constrained")
    ax2.plot(x, y, "-", linewidth=1.2, marker=".", markersize=2,
             label="Mouvements pour atteindre la cible")
    if echecs:
        ax2.scatter(echecs, [max_pas + 1] * len(echecs),
                    marker="x", s=12, alpha=0.6, color="tab:red",
                    label="Échec (cible non atteinte)")
    ax2.set(
        title=(f"Test après chaque épisode — départ fixe "
               f"({angles_test[0]}°, {angles_test[1]}°)"),
        xlabel="Épisodes d'entraînement",
        ylabel="Mouvements jusqu'à la cible — sans exploration",
    )
    ax2.set_xlim(0, len(trajets_fixes))
    ax2.set_ylim(0, max_pas + 3)
    valeurs = list(range(0, max_pas + 1, 10))
    ax2.set_yticks(valeurs + [max_pas + 1],
                   [str(i) for i in valeurs] + ["Échec"])
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Réussite globale tous les 500 épisodes et test fixe à chaque épisode."
    )
    parser.add_argument("--mode", choices=tuple(MODES_BRAS), default="distance")
    parser.add_argument("--actions", type=int, choices=(4, 9), default=9)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--angles-test", nargs=2, type=int,
                        default=ANGLES_TEST, metavar=("THETA1", "THETA2"))
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes doit être positif.")
    try:
        evaluations, trajets, max_pas = suivre(
            args.mode, args.actions, args.episodes, args.seed,
            angles_test=tuple(args.angles_test)
        )
    except ValueError as erreur:
        parser.error(str(erreur))
    tracer(evaluations, trajets, max_pas, args.mode, args.actions,
           tuple(args.angles_test))


if __name__ == "__main__":
    main()
