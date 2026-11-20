"""Entraînement d'un Deep Q-Network pour le bras robotique.

La Q-table est remplacée par un réseau PyTorch. Les tests utilisent
les mêmes indicateurs globaux que le Q-learning : taux de réussite et
coût moyen pénalisé sur toutes les positions de départ (échec = 51).

Depuis la racine du dépôt :
    py -m experiences.dqn_bras --episodes 5000
    py -m experiences.dqn_bras --episodes 5000 --eval-tous 1
    py -m experiences.dqn_bras --episodes 5000 --animer
"""

import argparse

import torch

from experiences.animation_bras import (
    MODES_BRAS, animer, position_depart, trajectoire_q,
)
from experiences.suivi_apprentissage import (
    evaluer_globalement, preparer_transitions, tracer,
)
from src.algorithmes.dqn import DQN


def suivre_dqn(mode="distance", actions=9, episodes=5000, seed=42,
                eval_tous=1, charger=None, sauvegarder=None):
    """Entraîner et évaluer le réseau sur toutes les positions initiales."""
    if episodes <= 0 or eval_tous <= 0:
        raise ValueError("episodes et eval_tous doivent être positifs.")

    # Réseau de petite taille sur CPU : un thread évite le surcoût du parallélisme.
    torch.set_num_threads(1)
    env = MODES_BRAS[mode](depart_aleatoire=True, seed=seed, nb_actions=actions)
    agent = DQN(env, nb_episodes=episodes, seed=seed)
    transitions = preparer_transitions(env)
    evaluations = []

    def mesurer(numero):
        # Test sans exploration ni apprentissage sur toutes les positions.
        reussis, total, taux, cout = evaluer_globalement(agent, transitions)
        evaluations.append((numero, taux, cout))
        # Éviter 5 000 lignes dans le terminal si on teste chaque épisode.
        if numero == 0 or numero % 500 == 0 or numero == episodes:
            print(f"{numero:>7} | {reussis:>4}/{total:<4} | "
                  f"{taux:>7.2f} % | {cout:>8.2f} | {agent.epsilon:.3f}")

    print(f"DQN — {actions} actions — {mode} — {episodes} épisodes")
    print("Épisode | Réussites   | Taux global | Coût pénalisé | Epsilon")
    print("-" * 74)

    if charger is not None:
        # Modèle déjà entraîné : évaluer sans recommencer l'apprentissage.
        agent.charger(charger, mode)
        mesurer(0)
    else:
        mesurer(0)
        dernier = 0

        def observer(_mouvements, _reussi):
            nonlocal dernier
            dernier += 1
            if dernier % eval_tous == 0 or dernier == episodes:
                mesurer(dernier)

        agent.apprendre(suivi_episodes=observer)
        if sauvegarder is not None:
            agent.sauvegarder(sauvegarder, mode)
            print(f"Modèle DQN enregistré : {sauvegarder}")

    return agent, evaluations


def main():
    parser = argparse.ArgumentParser(description="DQN sur bras robotique (PyTorch).")
    parser.add_argument("--mode", choices=tuple(MODES_BRAS), default="distance")
    parser.add_argument("--actions", type=int, choices=(4, 9), default=9)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-tous", type=int, default=1,
                        help="Tester tous les départs tous les N épisodes (défaut : 1).")
    parser.add_argument("--charger", help="Charger les poids d'un réseau déjà entraîné.")
    parser.add_argument("--sauvegarder", help="Enregistrer les poids DQN (.pt).")
    parser.add_argument("--sans-graphes", action="store_true")
    parser.add_argument("--animer", action="store_true")
    parser.add_argument("--depart", choices=("fixe", "aleatoire"), default="fixe")
    parser.add_argument("--angles", nargs=2, type=int, metavar=("THETA1", "THETA2"))
    parser.add_argument("--gif", help="Fichier de sortie GIF ; active l'animation.")
    parser.add_argument("--sans-fenetre", action="store_true",
                        help="Pas de fenêtre d'animation (requiert --gif).")
    args = parser.parse_args()

    if args.charger and args.sauvegarder:
        parser.error("--charger et --sauvegarder ne s'utilisent pas ensemble.")
    if args.angles and args.depart == "aleatoire":
        parser.error("--angles et --depart aleatoire sont incompatibles.")
    if args.sans_fenetre and not args.gif:
        parser.error("--sans-fenetre nécessite --gif.")
    if args.episodes <= 0 or args.eval_tous <= 0:
        parser.error("--episodes et --eval-tous doivent être positifs.")

    agent, mesures = suivre_dqn(args.mode, args.actions, args.episodes,
                               args.seed, args.eval_tous,
                               args.charger, args.sauvegarder)

    # L'animation existante n'utilise que agent.meilleure_action(etat).
    # Elle peut donc rejouer une politique DQN sans aucune Q-table.
    if args.animer or args.gif:
        depart = position_depart(agent.env, args.depart, args.angles, args.seed)
        etats, actions, reussi, raison = trajectoire_q(agent, depart,
                                                       max_pas=agent.max_pas)
        print(f"Animation DQN : départ {agent.env.angles(depart)}, "
              f"{len(actions)} mouvements, {raison}")
        animer(agent, etats, actions, reussi, raison,
               gif=args.gif, afficher=not args.sans_fenetre)

    if not args.sans_graphes:
        tracer(mesures, agent.max_pas, f"DQN / {args.mode}", args.actions,
               tous_les_points=True)


if __name__ == "__main__":
    main()
