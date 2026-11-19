"""Tracer la reussite au cours des episodes d'ENTRAINEMENT.

Cas etudie : epsilon=0, position initiale FIXE (180 degres, 90 degres).
Depuis cette pose, le plus court chemin vers la cible necessite 18 actions
(contre 9 depuis (90, 0)), sur le bras a 9 actions.
Aucune evaluation sur d'autres positions : le succes est mesure a la FIN
de chaque episode d'entrainement, avec exactement le QLearning du projet.

Lancer depuis la racine du depot :
    py -m experiences.reussite_epsilon_zero_depart_fixe
    py -m experiences.reussite_epsilon_zero_depart_fixe --episodes 5000

Les valeurs par episode (reussite 0/1, couverture cumulative,
exploration effective) sont exportees dans un CSV. Aucun lissage, aucune
moyenne glissante. Les cartes de l'agent final sont egalement produites.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiences.graphes import config_bras, creer_agent, DOSSIER_RESULTATS
from experiences.entrainement_bras_robotique import MODES_BRAS
from src.visualisation import graphes as cartes


def entrainer_et_mesurer(nb_episodes=5000, mode="distance", actions=9,
                        seed=42, max_pas=50, angles=(180, 90),
                        retourner_details=False):
    """Une seule Q-table et un depart fixe pour TOUS les episodes.

    Appeler apprendre() avec nb_episodes=1 ne reinitialise pas Q :
    seul l'historique de ce bloc est remis a zero, comme dans le projet.
    Apres l'appel, l'etat de l'environnement est l'etat final du meme episode.
    """
    if nb_episodes <= 0 or max_pas <= 0:
        raise ValueError("episodes et max_pas doivent etre positifs.")
    config = config_bras(
        mode, actions, depart_aleatoire=False,
        epsilon=0.0, epsilon_min=0.0, decroissance=1.0,
        nb_episodes=1, max_pas=max_pas
    )
    # La position est choisie lors de la construction de l'environnement,
    # afin que reset() recommence EXACTEMENT ici a chaque episode.
    config["construire"] = lambda seed: MODES_BRAS[mode](
        depart_aleatoire=False, angles_initiaux=tuple(angles),
        seed=seed, nb_actions=actions,
    )
    agent, env = creer_agent(config, seed)
    assert env.angles(env.etat_initial) == tuple(angles)
    assert env.angles(env.reset()) == tuple(angles)
    assert not env.depart_aleatoire
    assert np.count_nonzero(agent.Q) == 0

    reussites = np.zeros(nb_episodes, dtype=int)
    couvertures = np.zeros(nb_episodes, dtype=float)
    explorations = np.zeros(nb_episodes, dtype=float)
    etats_non_terminaux = np.asarray(env.etats_depart, dtype=int)
    nb_couples = len(etats_non_terminaux) * env.nb_actions
    for i in range(nb_episodes):
        agent.apprendre()
        # Vraie fin de l'episode : aucune evaluation sur d'autres positions.
        reussites[i] = int(env.distance_cible(env.etat) <= env.tolerance)
        # Couverture cumulative EXACTE apres cet episode.
        couvertures[i] = (
            100.0 * np.count_nonzero(agent.mises_a_jour[etats_non_terminaux])
            / nb_couples
        )
        # Fraction des actions aleatoires du meme episode (non cumulee).
        explorations[i] = 100.0 * agent.taux_exploration[0]
        assert agent.epsilon == 0.0

    if retourner_details:
        return reussites, couvertures, explorations, agent, env
    return reussites


def creer_graphe(reussites, mode="distance", actions=9, dossier=None,
                 angles=(180, 90)):
    """Un point par episode : 0=echec, 1=reussite, sans aucun lissage."""
    if dossier is None:
        dossier = (DOSSIER_RESULTATS / "bras_robotique" /
                   "epsilon_zero_depart_fixe" / f"{actions}_actions" / mode)
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    episodes = np.arange(1, len(reussites) + 1)
    valeurs = np.asarray(reussites, dtype=int)
    if not np.isin(valeurs, (0, 1)).all():
        raise ValueError("La reussite par episode doit valoir 0 ou 1.")

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    # Succes et echecs bruts, sans moyenne, ni fenetre, ni interpolation lisse.
    # Chaque marqueur correspond a un episode, sans interpolation ni lissage.
    echecs = valeurs == 0
    succes = ~echecs
    ax.scatter(episodes[succes], valeurs[succes], s=12, marker="|",
               color="#1764B0", label="Episodes reussis", zorder=2)
    ax.scatter(episodes[echecs], valeurs[echecs], s=16, marker="|",
               color="#C9363C", label="Episodes echoues", zorder=3)
    ax.set(
        title=(f"Reussite reelle par episode : epsilon = 0, "
               f"depart fixe ({angles[0]} deg, {angles[1]} deg)"),
        xlabel="Episode d'entrainement",
        ylabel="Resultat de l'episode",
    )
    ax.set_yticks([0, 1], ["Echec (0)", "Reussite (1)"])
    ax.set(xlim=(1, len(reussites)), ylim=(-0.15, 1.15))
    ax.grid(alpha=0.25)
    ax.legend()
    image = dossier / "reussite_par_episode.png"
    fig.savefig(image, dpi=160)
    plt.close(fig)

    fichier_csv = dossier / "reussite_par_episode.csv"
    with fichier_csv.open("w", newline="", encoding="utf-8") as fichier:
        writer = csv.writer(fichier)
        writer.writerow(("episode", "reussite_episode_0_ou_1"))
        writer.writerows(zip(episodes, valeurs))
    return image, fichier_csv


def tracer_autres_graphes(couvertures, explorations, agent, env,
                         mode="distance", actions=9, angles=(180, 90)):
    """Tracer les mesures brutes et les cartes issues DU MEME entrainement."""
    dossier = (DOSSIER_RESULTATS / "bras_robotique" /
               "epsilon_zero_depart_fixe" / f"{actions}_actions" / mode)
    dossier.mkdir(parents=True, exist_ok=True)
    episodes = np.arange(1, len(couvertures) + 1)
    series = (
        ("4_couverture.png", couvertures, "Couverture des couples (etat, action)",
         "Couples essayes depuis le debut de l'entrainement (%)"),
        ("5_exploration_effective.png", explorations,
         "Exploration effective par episode",
         "Actions tirees aleatoirement pendant cet episode (%)"),
    )
    for fichier, valeurs, titre, ylabel in series:
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        # Les mesures sont EXACTES et relevees apres chaque episode.
        ax.plot(episodes, valeurs, linewidth=1, color="#1764B0")
        ax.set(
            title=(f"{titre} - epsilon = 0, "
                   f"depart fixe ({angles[0]} deg, {angles[1]} deg)"),
            xlabel="Episode d'entrainement", ylabel=ylabel,
        )
        ax.set_xlim(1, len(episodes))
        ax.set_ylim(0, max(1.0, float(np.max(valeurs)) * 1.05))
        ax.grid(alpha=0.25)
        fig.savefig(dossier / fichier, dpi=160)
        plt.close(fig)

    nom = (f"Bras robotique {mode} ({actions} actions), "
           f"epsilon=0, depart fixe {tuple(angles)}")
    cartes.graphe_cartes_q_brutes(agent, env, nom,
                                 dossier / "3_cartes_q_brutes.png")
    cartes.graphe_meilleure_action(agent, env, nom,
                                  dossier / "4_meilleure_action.png")
    cartes.graphe_visites(agent, env, nom,
                          dossier / "6_visites.png")
    cartes.graphe_passages_par_action(agent, env, nom,
                                     dossier / "6b_passages_par_action.png")
    return dossier


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--mode", choices=("distance", "smart", "naive"),
                        default="distance")
    parser.add_argument("--actions", type=int, choices=(4, 9), default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pas", type=int, default=50)
    parser.add_argument("--angles", nargs=2, type=int, default=[180, 90],
                        metavar=("THETA1", "THETA2"),
                        help="Angles initiaux fixes, en degres (defaut : 180 90).")
    args = parser.parse_args()

    reussites, couvertures, explorations, agent, env = entrainer_et_mesurer(
        nb_episodes=args.episodes, mode=args.mode, actions=args.actions,
        seed=args.seed, max_pas=args.max_pas, angles=tuple(args.angles),
        retourner_details=True
    )
    image, csv_path = creer_graphe(
        reussites, mode=args.mode, actions=args.actions,
        angles=tuple(args.angles)
    )
    dossier = tracer_autres_graphes(
        couvertures, explorations, agent, env, mode=args.mode,
        actions=args.actions, angles=tuple(args.angles)
    )
    # CSV exact, un enregistrement PAR EPISODE, non lisse.
    with (dossier / "mesures_par_episode.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fichier:
        writer = csv.writer(fichier)
        writer.writerow(("episode", "reussite_0_ou_1",
                         "couverture_cumulative_pourcent",
                         "exploration_episode_pourcent"))
        writer.writerows(zip(
            range(1, len(reussites) + 1),
            reussites, couvertures, explorations
        ))
    print(f"Depart fixe a chaque episode : {tuple(args.angles)} degres")
    print(f"7 graphes et mesures CSV dans : {dossier}")


if __name__ == "__main__":
    main()
