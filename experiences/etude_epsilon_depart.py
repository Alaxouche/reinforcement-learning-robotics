"""Étudier epsilon ET la position initiale du bras (jamais Q initiale).

Q(s,a)=0 au début de chaque entraînement indépendant ; Q est apprise ensuite.
Deux stratégies : depart fixe (90°, 0°), ou aleatoire a chaque episode.
Dans les deux cas epsilon part de 1 et decroit vers un plancher de 0.05.

Exemple :
    py -m experiences.etude_epsilon_depart
    py -m experiences.etude_epsilon_depart --episodes 200 --seeds 42
    py -m experiences.etude_epsilon_depart --mode naive --actions 4
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiences.graphes import config_bras, entrainer_avec_suivi, creer_agent, DOSSIER_RESULTATS


DECROISSANCES = {"rapide": 0.99, "moyenne": 0.997, "lente": 0.9995}
DEPARTS = {"fixe": False, "aleatoire": True}
EPSILON_INITIAL = 1.0
EPSILON_MIN = 0.05


def experience_unique(mode, nb_actions, profil, depart, seed,
                      nb_episodes, pas_evaluation, nb_tests):
    """Une expérience : une nouvelle Q-table nulle et un seul mode de départ."""
    config = config_bras(
        mode, nb_actions, depart_aleatoire=DEPARTS[depart],
        epsilon=EPSILON_INITIAL, epsilon_min=EPSILON_MIN,
        decroissance=DECROISSANCES[profil], nb_episodes=nb_episodes
    )
    # creer_agent initialise Q a zero pour toutes les experiences du projet.
    agent, env = creer_agent(config, seed)
    assert np.all(agent.Q == 0), "La Q-table doit commencer a zero partout."
    assert env.depart_aleatoire == DEPARTS[depart]
    # L'entrainement repart d'une Q-table nulle, avec la meme graine.
    agent, env, evaluations = entrainer_avec_suivi(
        config, seed, pas_evaluation=pas_evaluation, nb_tests=nb_tests
    )
    return {
        "evaluations": evaluations,
        "epsilons": agent.epsilons,
        "exploration_effective": agent.taux_exploration,
        "couverture_etats": 100.0 * np.count_nonzero(agent.visites) / env.nb_etats,
    }


def etudier(mode="distance", nb_actions=9, profils=tuple(DECROISSANCES),
            seeds=(42, 43, 44), nb_episodes=5000, pas_evaluation=250,
            nb_tests=30):
    """Renvoie resultats[depart][profil], moyennes et ecarts-types des graines."""
    if not seeds or nb_episodes <= 0 or pas_evaluation <= 0 or nb_tests <= 0:
        raise ValueError("Graines non vides, episodes, pas et tests positifs requis.")
    resultats = {}
    for depart in DEPARTS:
        resultats[depart] = {}
        for profil in profils:
            essais = [
                experience_unique(mode, nb_actions, profil, depart, seed,
                                  nb_episodes, pas_evaluation, nb_tests)
                for seed in seeds
            ]
            nb_points = len(essais[0]["evaluations"])
            bilans = []
            for i in range(nb_points):
                point = {"episode": essais[0]["evaluations"][i]["episode"]}
                for cle in ("taux_reussite", "couples_essayes"):
                    valeurs = [essai["evaluations"][i][cle] for essai in essais]
                    point[cle] = float(np.mean(valeurs))
                    point[cle + "_std"] = float(np.std(valeurs))
                bilans.append(point)
            exploration = np.mean(
                np.array([essai["exploration_effective"] for essai in essais]),
                axis=0,
            )
            couverture_etats = float(np.mean([e["couverture_etats"] for e in essais]))
            dernier = bilans[-1]
            resultats[depart][profil] = {
                "evaluations": bilans,
                "epsilons": essais[0]["epsilons"],
                "exploration_effective": exploration.tolist(),
                "couverture_etats": couverture_etats,
            }
            print(
                f"{depart:>9} | {profil:>7} | epsilon_final="
                f"{essais[0]['epsilons'][-1]:.3f} | couples (s,a)="
                f"{dernier['couples_essayes']:.1f}% | reussite gloutonne="
                f"{dernier['taux_reussite']:.1f}% | etats visites="
                f"{couverture_etats:.1f}%"
            )
    return resultats


def tracer(resultats, dossier):
    """Quatre figures séparées ; ne représente jamais de résultats fictifs."""
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    for cle, titre, fichier in (
        ("couples_essayes", "Couverture des couples (etat, action)", "couverture.png"),
        ("taux_reussite", "Reussite de la politique gloutonne sans exploration", "reussite.png"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True,
                                 layout="constrained")
        for ax, depart in zip(axes, DEPARTS):
            for profil, donnees in resultats[depart].items():
                points = donnees["evaluations"]
                x = np.array([p["episode"] for p in points])
                y = np.array([p[cle] for p in points])
                std = np.array([p[cle + "_std"] for p in points])
                ligne, = ax.plot(x, y, label=profil)
                ax.fill_between(x, np.clip(y - std, 0, 100),
                                np.clip(y + std, 0, 100),
                                color=ligne.get_color(), alpha=0.15)
            ax.set_title("Depart " + depart)
            ax.set_xlabel("Episodes")
            ax.grid(alpha=0.25)
            ax.set_ylim(0, 105)
        axes[0].set_ylabel("Pourcentage (%)")
        axes[1].legend(title="Decroissance")
        fig.suptitle(titre + " (moyenne ± ecart-type entre graines)")
        fig.savefig(dossier / fichier, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for profil, donnees in resultats["fixe"].items():
        ax.plot(np.arange(1, len(donnees["epsilons"]) + 1),
                donnees["epsilons"], label=profil)
    ax.set(xlabel="Episode", ylabel="Epsilon", title="Strategies de decroissance d'epsilon")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(dossier / "epsilon.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True,
                             layout="constrained")
    for ax, depart in zip(axes, DEPARTS):
        for profil, donnees in resultats[depart].items():
            # Moyenne glissante de 100 episodes pour la lisibilite.
            y = np.array(donnees["exploration_effective"], dtype=float)
            fenetre = min(100, len(y))
            x = np.arange(fenetre, len(y) + 1)
            moyenne = np.convolve(y, np.ones(fenetre) / fenetre, "valid") * 100
            ax.plot(x, moyenne, label=profil)
        ax.set(xlabel="Episode", title="Depart " + depart)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Actions tirees au hasard (%)")
    axes[1].legend(title="Decroissance")
    fig.suptitle("Exploration effective pendant l'entrainement")
    fig.savefig(dossier / "exploration_effective.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Influence de la decroissance d'epsilon : bras fixe versus aleatoire, Q=0."
    )
    parser.add_argument("--mode", choices=("naive", "smart", "distance"), default="distance")
    parser.add_argument("--actions", type=int, choices=(4, 9), default=9)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--profils", nargs="+", choices=tuple(DECROISSANCES),
                        default=list(DECROISSANCES))
    parser.add_argument("--pas-evaluation", type=int, default=250)
    parser.add_argument("--tests", type=int, default=30)
    args = parser.parse_args()

    resultats = etudier(
        mode=args.mode, nb_actions=args.actions, profils=tuple(args.profils),
        seeds=tuple(args.seeds), nb_episodes=args.episodes,
        pas_evaluation=args.pas_evaluation, nb_tests=args.tests,
    )
    dossier = (DOSSIER_RESULTATS / "bras_robotique" / "epsilon_depart"
               / f"{args.actions}_actions" / args.mode)
    tracer(resultats, dossier)
    print(f"Graphiques enregistres dans {dossier}")


if __name__ == "__main__":
    main()
