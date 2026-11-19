"""Tracer la reussite au cours des episodes d'ENTRAINEMENT.

Cas etudie : epsilon=0, position initiale FIXE (90 degres, 0 degre).
Aucune evaluation sur d'autres positions : le succes est mesure a la FIN
de chaque episode d'entrainement, avec exactement le QLearning du projet.

Lancer depuis la racine du depot :
    py -m experiences.reussite_epsilon_zero_depart_fixe
    py -m experiences.reussite_epsilon_zero_depart_fixe --episodes 5000 --fenetre 100

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


def entrainer_et_mesurer(nb_episodes=5000, mode="distance", actions=9,
                        seed=42, max_pas=50):
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
    agent, env = creer_agent(config, seed)
    assert env.angles(env.etat_initial) == (90, 0)
    assert not env.depart_aleatoire
    assert np.count_nonzero(agent.Q) == 0

    reussites = np.zeros(nb_episodes, dtype=int)
    for i in range(nb_episodes):
        agent.apprendre()
        # Le vrai etat final de cet episode, sans evaluation supplementaire.
        reussites[i] = int(env.distance_cible(env.etat) <= env.tolerance)
        assert agent.epsilon == 0.0

    return reussites


def moyenne_glissante(reussites, fenetre=100):
    """Pourcentage sur les min(fenetre, episode) derniers episodes."""
    if fenetre <= 0:
        raise ValueError("La fenetre doit etre positive.")
    reussites = np.asarray(reussites, dtype=int)
    episodes = np.arange(1, len(reussites) + 1)
    debut = np.maximum(episodes - fenetre, 0)
    cumul = np.concatenate(([0], np.cumsum(reussites)))
    return 100.0 * (cumul[episodes] - cumul[debut]) / (episodes - debut)


def creer_graphe(reussites, fenetre=100, mode="distance", actions=9, dossier=None):
    """Une figure : taux de reussite par episode, sans departs de test."""
    if dossier is None:
        dossier = (DOSSIER_RESULTATS / "bras_robotique" /
                   "epsilon_zero_depart_fixe" / f"{actions}_actions" / mode)
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    taux = moyenne_glissante(reussites, fenetre)
    episodes = np.arange(1, len(reussites) + 1)

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.plot(episodes, taux, color="#1764B0", linewidth=2)
    ax.set(
        title="Reussite pendant l'entrainement : epsilon = 0, depart fixe (90 deg, 0 deg)",
        xlabel="Episode d'entrainement",
        ylabel=f"Reussite sur les {fenetre} derniers episodes (%)",
    )
    ax.set(xlim=(1, len(reussites)), ylim=(-2, 102))
    ax.grid(alpha=0.25)
    image = dossier / "reussite_par_episode.png"
    fig.savefig(image, dpi=160)
    plt.close(fig)

    fichier_csv = dossier / "reussite_par_episode.csv"
    with fichier_csv.open("w", newline="", encoding="utf-8") as fichier:
        writer = csv.writer(fichier)
        writer.writerow(("episode", "reussite_episode_0_ou_1",
                         "reussite_glissante_pourcent"))
        writer.writerows(zip(episodes, reussites, taux))
    return image, fichier_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--fenetre", type=int, default=100)
    parser.add_argument("--mode", choices=("distance", "smart", "naive"),
                        default="distance")
    parser.add_argument("--actions", type=int, choices=(4, 9), default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pas", type=int, default=50)
    args = parser.parse_args()

    reussites = entrainer_et_mesurer(
        nb_episodes=args.episodes, mode=args.mode, actions=args.actions,
        seed=args.seed, max_pas=args.max_pas
    )
    image, csv_path = creer_graphe(
        reussites, fenetre=args.fenetre, mode=args.mode, actions=args.actions
    )
    print(f"Graphique : {image}")
    print(f"Donnees par episode : {csv_path}")


if __name__ == "__main__":
    main()
