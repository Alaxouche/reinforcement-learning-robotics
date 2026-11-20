"""Évolution du Q-learning sur TOUS les départs possibles du bras robotique.

Entraînement : un départ aléatoire et epsilon-greedy à chaque épisode.
Après CHAQUE épisode, évaluation de la politique argmax(Q) sur toutes les
configurations de départ non terminales. Aucune exploration, aucune mise à jour
de Q pendant le test. Le résultat est un coût moyen en mouvements, avec
51 pour chaque échec si max_pas=50 : ce n'est pas une moyenne conditionnée
aux seules réussites.

Les transitions déterministes sont mises en cache et les trajets partagent
leurs résultats : on évite de rejouer plusieurs millions de trajets complets.
Pas de BFS, pas de moyenne glissante, pas de dossier de résultats.
"""

import argparse

import matplotlib.pyplot as plt

from experiences.animation_bras import MODES_BRAS, creer_agent

PAS = 500


def preparer_transitions(env):
    """Préparer toutes les transitions déterministes du bras, une seule fois.

    On reproduit exactement les rotations modulo 360 de BrasRobotique2D.step().
    Dans ce bras sans butées, toutes les actions sont autorisées hors cible.
    """
    n1 = len(env.angles_theta1)
    n2 = env.nb_theta2
    transitions = []
    for etat in range(env.nb_etats):
        i, j = divmod(etat, n2)
        transitions.append(tuple(
            ((i + di) % n1) * n2 + ((j + dj) % n2)
            for di, dj in env.deplacements
        ))
    return transitions


def evaluer_globalement(agent, transitions):
    """Tester tous les départs avec argmax(Q), sans modifier l'environnement.

    Chaque état a un seul successeur sous la politique gloutonne. On mémorise
    le nombre de pas jusqu'à la cible et détecte les cycles. Un chemin qui
    échoue ou dépasse max_pas reçoit le coût pénalisé max_pas+1.
    """
    env = agent.env
    limite = agent.max_pas
    penalite = limite + 1
    # Pour le DQN, les actions sont prédites par le réseau ; aucune Q-table.
    # Le Q-learning tabulaire conserve son comportement d'origine.
    action_gloutonne = (agent.actions_gloutonnes()
                        if hasattr(agent, "actions_gloutonnes")
                        else agent.Q.argmax(axis=1))

    successeurs = [
        transitions[etat][int(action_gloutonne[etat])]
        for etat in range(env.nb_etats)
    ]

    INCONNU = -1
    distances = [INCONNU] * env.nb_etats
    for etat_cible in env.etats_objectif:
        distances[etat_cible] = 0

    for depart in env.etats_depart:
        if distances[depart] != INCONNU:
            continue

        chemin = []
        vus = set()
        etat = depart
        while distances[etat] == INCONNU and etat not in vus:
            vus.add(etat)
            chemin.append(etat)
            etat = successeurs[etat]

        # Une cible donne 0, une boucle donne la pénalité.
        longueur = (distances[etat]
                    if distances[etat] != INCONNU else penalite)
        for ancien in reversed(chemin):
            longueur = min(longueur + 1, penalite)
            distances[ancien] = longueur

    departs = env.etats_depart
    reussites = sum(distances[s] <= limite for s in departs)
    cout_moyen = sum(distances[s] for s in departs) / len(departs)
    return reussites, len(departs), 100.0 * reussites / len(departs), cout_moyen


def suivre(mode="distance", actions=9, episodes=5000, seed=42):
    """Entraîner normalement et évaluer tous les départs après chaque épisode."""
    if episodes <= 0:
        raise ValueError("Le nombre d'épisodes doit être positif.")

    agent = creer_agent(
        mode=mode, nb_actions=actions, episodes=episodes,
        seed=seed, entrainer=False
    )
    transitions = preparer_transitions(agent.env)
    evaluations = []  # (épisode, taux, coût moyen pénalisé)

    def mesurer(numero, afficher=False):
        reussites, total, taux, cout = evaluer_globalement(agent, transitions)
        evaluations.append((numero, taux, cout))
        if afficher:
            print(f"{numero:>7} | {reussites:>4}/{total:<4} | "
                  f"{taux:>7.2f} % | {cout:>8.2f}")

    print(f"Bras {actions} actions | récompense {mode} | "
          f"{episodes} épisodes | epsilon dynamique habituel")
    print("Test : tous les départs, sans exploration | échec = "
          f"{agent.max_pas + 1} points")
    print("Épisode | Réussites   | Taux global | Coût moyen pénalisé")
    print("-" * 70)

    mesurer(0, afficher=True)
    try:
        for debut in range(0, episodes, PAS):
            agent.nb_episodes = min(PAS, episodes - debut)

            def observer(_mouvements_entrainement, _reussi_entrainement):
                numero = len(evaluations)
                mesurer(numero, afficher=False)

            agent.apprendre(suivi_episodes=observer)
            # Évaluation déjà effectuée après chacun des épisodes du bloc.
            numero = len(evaluations) - 1
            print(f"{numero:>7} | "
                  f"{round(evaluations[-1][1] * len(agent.env.etats_depart) / 100):>4}"
                  f"/{len(agent.env.etats_depart):<4} | "
                  f"{evaluations[-1][1]:>7.2f} % | "
                  f"{evaluations[-1][2]:>8.2f}")
    finally:
        agent.nb_episodes = episodes

    return evaluations, agent.max_pas


def tracer(evaluations, max_pas, mode, actions, tous_les_points=False):
    """Deux séries d'observations sans lissage, avec la même Q-table testée."""
    numeros = [p[0] for p in evaluations]
    taux = [p[1] for p in evaluations]
    couts = [p[2] for p in evaluations]

    # Pour rester lisible, le taux de réussite est affiché tous les 500 épisodes.
    # DQN : afficher chaque mesure disponible ; Q-learning : tous les 500.
    points = (evaluations if tous_les_points else
              [p for p in evaluations if p[0] % PAS == 0 or p[0] == numeros[-1]])
    fig1, ax1 = plt.subplots(figsize=(9, 5), layout="constrained")
    ax1.plot([p[0] for p in points], [p[1] for p in points],
             "-o", linewidth=2)
    ax1.set(title=f"Réussite sur tous les départs — {actions} actions — {mode}",
            xlabel="Épisodes d'entraînement",
            ylabel="Départs atteignant la cible (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(alpha=0.3)

    fig2, ax2 = plt.subplots(figsize=(10, 5), layout="constrained")
    ax2.plot(numeros, couts, linewidth=1.3)
    ax2.set(
        title=f"Coût moyen global après chaque épisode — {actions} actions — {mode}",
        xlabel="Épisodes d'entraînement",
        ylabel=f"Mouvements moyens pénalisés (échec = {max_pas + 1})",
    )
    ax2.set_xlim(0, max(1, numeros[-1]))  # un seul point si modèle rechargé
    ax2.set_ylim(0, max_pas + 2)
    ax2.grid(alpha=0.3)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Évaluer tous les départs après chaque épisode Q-learning."
    )
    parser.add_argument("--mode", choices=tuple(MODES_BRAS), default="distance")
    parser.add_argument("--actions", type=int, choices=(4, 9), default=9)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes doit être positif.")
    evaluations, max_pas = suivre(args.mode, args.actions, args.episodes, args.seed)
    tracer(evaluations, max_pas, args.mode, args.actions)


if __name__ == "__main__":
    main()
