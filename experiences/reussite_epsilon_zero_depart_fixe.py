"""Tracer la reussite au cours des episodes d'ENTRAINEMENT.

Cas etudie : epsilon=0, position initiale FIXE (180 degres, 90 degres).
Depuis cette pose, le plus court chemin vers la cible necessite 18 actions
(contre 9 depuis (90, 0)), sur le bras a 9 actions.
Aucune evaluation sur d'autres positions : le succes est mesure a la FIN
de chaque episode d'entrainement, avec exactement le QLearning du projet.

Lancer depuis la racine du depot :
    py -m experiences.reussite_epsilon_zero_depart_fixe
    py -m experiences.reussite_epsilon_zero_depart_fixe --episodes 5000

Les donnees brutes (0 ou 1 par episode) sont aussi exportees en CSV.
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


def entrainer_et_mesurer(nb_episodes=5000, mode="distance", actions=9,
                        seed=42, max_pas=50, angles=(180, 90)):
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
    for i in range(nb_episodes):
        agent.apprendre()
        # Le vrai etat final de cet episode, sans evaluation supplementaire.
        reussites[i] = int(env.distance_cible(env.etat) <= env.tolerance)
        assert agent.epsilon == 0.0

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
    ax.step(episodes, valeurs, where="mid", linewidth=0.7, color="#1764B0")
    echecs = valeurs == 0
    ax.scatter(episodes[echecs], valeurs[echecs], s=8, color="#C9363C",
               label="Episodes echoues", zorder=3)
    ax.set(
        title=(f"Reussite reelle par episode : epsilon = 0, "
               f"depart fixe ({angles[0]} deg, {angles[1]} deg)"),
        xlabel="Episode d'entrainement",
        ylabel="Resultat de l'episode",
    )
    ax.set_yticks([0, 1], ["Echec (0)", "Reussite (1)"])
    ax.set(xlim=(1, len(reussites)), ylim=(-0.15, 1.15))
    ax.grid(alpha=0.25)
    if np.any(echecs):
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

    reussites = entrainer_et_mesurer(
        nb_episodes=args.episodes, mode=args.mode, actions=args.actions,
        seed=args.seed, max_pas=args.max_pas, angles=tuple(args.angles)
    )
    image, csv_path = creer_graphe(
        reussites, mode=args.mode, actions=args.actions,
        angles=tuple(args.angles)
    )
    print(f"Depart fixe a chaque episode : {tuple(args.angles)} degres")
    print(f"Graphique : {image}")
    print(f"Donnees par episode : {csv_path}")


if __name__ == "__main__":
    main()
