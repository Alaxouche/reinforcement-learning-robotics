"""Comparer epsilon fixe et decroissant sur la qualite des trajets appris.

Depuis la racine :
    py -m experiences.comparaison_epsilon
    py -m experiences.comparaison_epsilon --tests 0  # tous les departs (plus long)

Meme Q nulle, memes graines, memes departs de test et meme nombre d'episodes.
Deux metriques : part des chemins optimaux et detour moyen parmi les reussites.
"""

import argparse
import csv
import random
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiences.graphes import DOSSIER_RESULTATS, config_bras, creer_agent
from experiences.entrainement_bras_robotique import suivre_politique


STRATEGIES = {
    "fixe 0.05": (0.05, 0.05, 1.0),
    "fixe 0.2": (0.2, 0.2, 1.0),
    "fixe 0.5": (0.5, 0.5, 1.0),
    "dynamique": (1.0, 0.05, 0.9995),
}
COULEURS = ("#C9363C", "#8B43B5", "#DF8500", "#1764B0")


def distances_minimales(env):
    """BFS inverse : nb minimal d'actions jusqu'a une pose cible.

    Les articulations tournent modulo 360. Tous les mouvements autorises
    ont un cout de 1 ; l'action immobile (si presente) ne change rien.
    """
    n1 = len(env.angles_theta1)
    n2 = env.nb_theta2
    distances = np.full(env.nb_etats, -1, dtype=int)
    file = deque(env.etats_objectif)
    for etat in env.etats_objectif:
        distances[etat] = 0

    while file:
        etat = file.popleft()
        i, j = divmod(etat, n2)
        for di, dj in env.deplacements:
            precedent = ((i - di) % n1) * n2 + ((j - dj) % n2)
            if distances[precedent] < 0:
                distances[precedent] = distances[etat] + 1
                file.append(precedent)
    if np.any(distances < 0):
        raise ValueError("Certains etats n'ont pas de chemin vers la cible.")
    return distances


def evaluer_trajets(agent, departs_test, distances):
    """Evaluation gloutonne ; epsilon ne sert pas pendant l'evaluation."""
    env = agent.env
    etat_precedent = env.etat
    nb_reussites = 0
    nb_optimaux = 0
    detours = []
    try:
        for depart in departs_test:
            parcours, reussi = suivre_politique(agent, int(depart))
            if reussi:
                nb_reussites += 1
                longueur = len(parcours) - 1
                optimum = int(distances[depart])
                detour = longueur - optimum
                if detour < 0:
                    raise AssertionError("Longueur inferieure a l'optimum BFS.")
                detours.append(detour)
                if detour == 0:
                    nb_optimaux += 1
    finally:
        env.etat = etat_precedent
    return {
        "taux_reussite": 100.0 * nb_reussites / len(departs_test),
        "taux_optimal": 100.0 * nb_optimaux / len(departs_test),
        "detour_moyen": float(np.mean(detours)) if detours else float("nan"),
    }


def entrainer_strategie(nom, parametres, seed, mode, actions, depart,
                       episodes, pas, tests):
    epsilon, epsilon_min, decroissance = parametres
    config = config_bras(
        mode, actions, depart_aleatoire=(depart == "aleatoire"),
        epsilon=epsilon, epsilon_min=epsilon_min,
        decroissance=decroissance, nb_episodes=episodes,
    )
    agent, env = creer_agent(config, seed)
    assert np.all(agent.Q == 0)
    distances = distances_minimales(env)
    if tests == 0:
        departs_test = env.etats_depart
    else:
        departs_test = random.Random(seed + 10_000).sample(
            env.etats_depart, k=min(tests, len(env.etats_depart))
        )

    points = []
    try:
        for debut in range(0, episodes, pas):
            agent.nb_episodes = min(pas, episodes - debut)
            agent.apprendre()
            mesure = evaluer_trajets(agent, departs_test, distances)
            points.append({
                "strategie": nom, "seed": seed,
                "episode": debut + agent.nb_episodes,
                "epsilon_effectif": float(agent.epsilon),
                "n_tests": len(departs_test),
                **mesure,
            })
    finally:
        agent.nb_episodes = episodes
    return points


def tracer(resultats, cle, titre, axe_y, chemin):
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for (nom, essais), couleur in zip(resultats.items(), COULEURS):
        x = [p["episode"] for p in essais[0]]
        valeurs = np.asarray([[p[cle] for p in essai] for essai in essais],
                             dtype=float)
        moyennes = []
        ecarts = []
        for col in valeurs.T:
            valides = col[np.isfinite(col)]
            moyennes.append(float(np.mean(valides)) if len(valides) else np.nan)
            ecarts.append(float(np.std(valides)) if len(valides) else np.nan)
        y = np.asarray(moyennes)
        ecart = np.asarray(ecarts)
        ax.plot(x, y, marker="o", markersize=2.5,
                color=couleur, label=nom)
        ax.fill_between(x, y - ecart, y + ecart,
                        alpha=0.12, color=couleur)
    ax.set(title=titre, xlabel="Episodes", ylabel=axe_y)
    if cle == "taux_optimal":
        ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def comparer(mode="distance", actions=9, depart="aleatoire",
            episodes=5000, pas=250, tests=100, seeds=(42, 43, 44)):
    if episodes <= 0 or pas <= 0 or tests < 0 or not seeds:
        raise ValueError("Parametres invalides : episodes, pas, tests, seeds.")

    resultats = {}
    for nom, parametres in STRATEGIES.items():
        resultats[nom] = [
            entrainer_strategie(nom, parametres, seed, mode, actions,
                                depart, episodes, pas, tests)
            for seed in seeds
        ]

    dossier = (DOSSIER_RESULTATS / "bras_robotique" / "comparaison_epsilon"
               / f"{actions}_actions" / mode / depart)
    dossier.mkdir(parents=True, exist_ok=True)
    tracer(resultats, "taux_optimal",
           "Chemins de longueur minimale (BFS)",
           "Departs avec trajet optimal (%)",
           dossier / "1_chemins_optimaux.png")
    tracer(resultats, "detour_moyen",
           "Mouvements supplementaires sur les trajets reussis",
           "Detour moyen par rapport au BFS (actions)",
           dossier / "2_detours.png")

    with (dossier / "mesures.csv").open("w", newline="", encoding="utf-8") as f:
        champs = ("strategie", "seed", "episode", "epsilon_effectif",
                  "n_tests", "taux_reussite", "taux_optimal", "detour_moyen")
        ecrivain = csv.DictWriter(f, fieldnames=champs)
        ecrivain.writeheader()
        for essais in resultats.values():
            for essai in essais:
                ecrivain.writerows(essai)
    print(f"2 graphes et mesures.csv enregistres dans : {dossier}")
    return resultats


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("naive", "smart", "distance"),
                        default="distance")
    parser.add_argument("--actions", type=int, choices=(4, 9), default=9)
    parser.add_argument("--depart", choices=("fixe", "aleatoire"),
                        default="aleatoire")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--pas-evaluation", type=int, default=250)
    parser.add_argument("--tests", type=int, default=100,
                        help="Nombre de departs evalues ; 0 = tous les departs.")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 43, 44])
    args = parser.parse_args()
    comparer(mode=args.mode, actions=args.actions, depart=args.depart,
             episodes=args.episodes, pas=args.pas_evaluation,
             tests=args.tests, seeds=tuple(args.seeds))


if __name__ == "__main__":
    main()
