"""Comparer uniquement deux indicateurs d'exploration pour epsilon constant.

Lancement depuis la racine : py -m experiences.etude_epsilon_fixe

Chaque (epsilon, graine) repart d'une Q-table nulle. Les figures conservees :
4_couverture.png et 5_exploration_effective.png. La reussite, la variation
de Q et le bilan final ne sont plus traces dans cette experience.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiences.graphes import DOSSIER_RESULTATS, config_bras, creer_agent


EPSILONS_PAR_DEFAUT = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)


def experience(epsilon, seed, mode, actions, depart, episodes, pas, tests):
    """Uniquement la couverture des couples et l'exploration effective."""
    config = config_bras(
        mode, actions, depart_aleatoire=(depart == "aleatoire"),
        epsilon=epsilon, epsilon_min=epsilon, decroissance=1.0,
        nb_episodes=episodes,
    )
    agent, env = creer_agent(config, seed)
    assert np.all(agent.Q == 0)
    couples = len(env.etats_depart) * env.nb_actions
    historique = []
    try:
        for debut in range(0, episodes, pas):
            agent.nb_episodes = min(pas, episodes - debut)
            agent.apprendre()
            assert np.allclose(agent.epsilons, epsilon)
            historique.append({
                "epsilon": epsilon, "seed": seed,
                "episode": debut + agent.nb_episodes,
                "couverture": 100.0 * np.count_nonzero(
                    agent.mises_a_jour[env.etats_depart]
                ) / couples,
                "exploration_effective": 100.0 * np.mean(agent.taux_exploration),
            })
    finally:
        agent.nb_episodes = episodes
    return historique


def agreger(essais, champ):
    """Moyenne et ecart-type inter-graines, dans l'ordre des evaluations."""
    valeurs = np.array([
        [np.nan if p[champ] is None else p[champ] for p in essai]
        for essai in essais
    ], dtype=float)
    episodes = np.array([p["episode"] for p in essais[0]], dtype=int)
    valides = ~np.all(np.isnan(valeurs), axis=0)
    x = episodes[valides]
    if not np.any(valides):
        return x, np.array([]), np.array([])
    y = valeurs[:, valides]
    return x, np.mean(y, axis=0), np.std(y, axis=0)


def tracer_courbe(resultats, champ, titre, ylabel, chemin, borne_100=False):
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")

    # Six couleurs explicites pour eviter qu'un style Matplotlib externe
    # ne recycle sa palette avant la fin de la comparaison.
    palette = ("#1764B0", "#C9363C", "#16865D",
               "#8B43B5", "#DF8500", "#008FA3")
    couleurs = {
        epsilon: palette[i] if i < len(palette)
        else plt.get_cmap("hsv")(i / len(resultats))
        for i, epsilon in enumerate(resultats)
    }
    for epsilon, essais in resultats.items():
        x, moyenne, ecart = agreger(essais, champ)
        if len(x) == 0:
            continue
        couleur = couleurs[epsilon]
        ax.plot(x, moyenne, marker="o", markersize=2.5,
                color=couleur, label=f"epsilon = {epsilon:g}")
        ax.fill_between(x, moyenne - ecart, moyenne + ecart,
                        alpha=0.12, color=couleur)
    ax.set(xlabel="Episodes", ylabel=ylabel, title=titre)
    if borne_100:
        ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(chemin, dpi=150)
    plt.close(fig)


def sauvegarder_csv(resultats, chemin):
    champs = (
        "epsilon", "seed", "episode", "couverture", "exploration_effective",
    )
    with chemin.open("w", newline="", encoding="utf-8") as fichier:
        ecrivain = csv.DictWriter(fichier, fieldnames=champs)
        ecrivain.writeheader()
        for essais in resultats.values():
            for essai in essais:
                ecrivain.writerows(essai)


def etudier(epsilons=EPSILONS_PAR_DEFAUT, seeds=(42, 43, 44),
            mode="distance", actions=9, depart="aleatoire",
            episodes=5000, pas=250, tests=100):
    if not epsilons or not seeds:
        raise ValueError("Prevoir au moins un epsilon et une graine.")
    if any(not np.isfinite(e) or not 0 <= e <= 1 for e in epsilons):
        raise ValueError("Chaque epsilon doit etre dans [0, 1].")
    if episodes <= 0 or pas <= 0 or tests <= 0:
        raise ValueError("episodes, pas et tests doivent etre positifs.")

    dossier = (DOSSIER_RESULTATS / "bras_robotique" / "epsilon_fixe"
               / f"{actions}_actions" / mode / depart)
    dossier.mkdir(parents=True, exist_ok=True)
    resultats = {}

    for epsilon in dict.fromkeys(epsilons):
        essais = []
        for seed in seeds:
            essais.append(experience(
                epsilon, seed, mode, actions, depart, episodes, pas, tests
            ))
        resultats[epsilon] = essais

    # Deux figures conservees : couverture et exploration effective.
    # Supprimer les anciens fichiers pour eviter toute confusion.
    for ancien in ("1_reussite.png", "2_stabilite_politique.png",
                   "3_variation_q.png", "6_bilan_final.png"):
        (dossier / ancien).unlink(missing_ok=True)
    tracer_courbe(resultats, "couverture",
                  "Couverture des couples (etat, action)",
                  "Couples essayes (%)", dossier / "4_couverture.png", True)
    tracer_courbe(resultats, "exploration_effective",
                  "Exploration effective pendant l'entrainement",
                  "Actions tirees au hasard (%)",
                  dossier / "5_exploration_effective.png", True)
    sauvegarder_csv(resultats, dossier / "mesures.csv")
    (dossier / "diagnostics.csv").unlink(missing_ok=True)
    print(f"2 graphes et mesures.csv enregistres dans : {dossier}")
    return resultats


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epsilons", nargs="+", type=float,
                        default=list(EPSILONS_PAR_DEFAUT))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--mode", choices=("naive", "smart", "distance"),
                        default="distance")
    parser.add_argument("--actions", type=int, choices=(4, 9), default=9)
    parser.add_argument("--depart", choices=("fixe", "aleatoire"),
                        default="aleatoire")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--pas-evaluation", type=int, default=250)
    parser.add_argument("--tests", type=int, default=100)
    args = parser.parse_args()
    etudier(epsilons=tuple(args.epsilons), seeds=tuple(args.seeds),
            mode=args.mode, actions=args.actions, depart=args.depart,
            episodes=args.episodes, pas=args.pas_evaluation, tests=args.tests)


if __name__ == "__main__":
    main()
