"""Comparer epsilon constant : reussite et couverture, departs fixes/aleatoires.

Produit exactement quatre figures :
    1_reussite_aleatoire.png
    2_couverture_aleatoire.png
    3_reussite_fixe.png
    4_couverture_fixe.png

Reussite : proportion EXACTE d'episodes reussis sur chaque bloc
NON CHEVAUCHANT (100 episodes par defaut), sans moyenne glissante.
Couverture : proportion cumulative EXACTE de couples (etat, action) essayes,
relevee apres chaque episode. Les reussites brutes 0/1 sont sauvegardees
en CSV et ne sont jamais remplacees par une evaluation sur des departs tests.

Lancer depuis la racine du depot :
    py -m experiences.comparaison_epsilon_reussite_couverture
    py -m experiences.comparaison_epsilon_reussite_couverture --angles-fixes 180 90
"""

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiences.entrainement_bras_robotique import MODES_BRAS
from experiences.graphes import DOSSIER_RESULTATS, config_bras, creer_agent


EPSILONS_PAR_DEFAUT = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
COULEURS = (
    "#1764B0", "#C9363C", "#16865D",
    "#8B43B5", "#DF8500", "#008FA3",
)


def entrainer_une_configuration(epsilon, depart_aleatoire, mode="distance",
                                actions=9, angles_fixes=(90, 0),
                                episodes=5000, bloc=100, seed=42,
                                max_pas=50):
    """Une Q-table neuve, epsilon fixe, resultats du VRAI entrainement.

    Chaque appel a agent.apprendre() lance un seul episode : la Q-table
    ET les compteurs de visites / mises a jour sont conserves entre appels.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon doit appartenir a [0, 1].")
    if episodes <= 0 or bloc <= 0 or max_pas <= 0:
        raise ValueError("episodes, bloc et max_pas doivent etre positifs.")

    config = config_bras(
        mode, actions, depart_aleatoire=depart_aleatoire,
        epsilon=float(epsilon), epsilon_min=float(epsilon),
        decroissance=1.0, nb_episodes=1, max_pas=max_pas,
    )
    if not depart_aleatoire:
        # Le depart fixe est configure dans le constructeur, pas impose
        # uniquement au premier episode.
        config["construire"] = lambda graine: MODES_BRAS[mode](
            depart_aleatoire=False, angles_initiaux=tuple(angles_fixes),
            seed=graine, nb_actions=actions,
        )

    agent, env = creer_agent(config, seed)
    if np.count_nonzero(agent.Q) != 0:
        raise AssertionError("La Q-table doit commencer a zero.")
    if env.depart_aleatoire != depart_aleatoire:
        raise AssertionError("Le mode de depart est incorrect.")
    if not depart_aleatoire:
        if env.angles(env.etat_initial) != tuple(angles_fixes):
            raise AssertionError("La position initiale fixe est incorrecte.")
        if env.etat_initial not in env.etats_depart:
            raise ValueError("La position initiale est deja sur la cible.")

    etats_autorises = np.asarray(env.etats_depart, dtype=int)
    nombre_couples = len(etats_autorises) * env.nb_actions
    reussites_brutes = np.empty(episodes, dtype=int)
    couverture_brute = np.empty(episodes, dtype=float)

    for i in range(episodes):
        agent.apprendre()
        # L'etat final est celui de l'episode d'entrainement lui-meme,
        # sans evaluation sur 100 autres positions.
        reussites_brutes[i] = int(env.distance_cible(env.etat) <= env.tolerance)
        couverture_brute[i] = (
            100.0 * np.count_nonzero(agent.mises_a_jour[etats_autorises])
            / nombre_couples
        )
        if agent.epsilon != epsilon:
            raise AssertionError("Epsilon doit rester constant.")

    # Les blocs ne se chevauchent pas ; il ne s'agit PAS d'une moyenne glissante.
    x_reussite = []
    y_reussite = []
    for debut in range(0, episodes, bloc):
        fin = min(debut + bloc, episodes)
        succes = int(np.sum(reussites_brutes[debut:fin]))
        x_reussite.append(fin)
        y_reussite.append(100.0 * succes / (fin - debut))

    return {
        "epsilon": epsilon,
        "episodes": np.arange(1, episodes + 1),
        "reussites_brutes": reussites_brutes,
        "couverture": couverture_brute,
        "x_reussite": np.asarray(x_reussite),
        "y_reussite": np.asarray(y_reussite),
    }


def tracer(resultats, mesure, depart, dossier, bloc, angles_fixes):
    est_reussite = mesure == "reussite"
    titre = "Reussite" if est_reussite else "Couverture des couples (etat, action)"
    qualification = ("departs aleatoires" if depart == "aleatoire"
                     else f"depart fixe {tuple(angles_fixes)}")
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")

    for indice, (epsilon, donnees) in enumerate(resultats.items()):
        couleur = (COULEURS[indice] if indice < len(COULEURS)
                   else plt.get_cmap("hsv")(indice / len(resultats)))
        if est_reussite:
            x, y = donnees["x_reussite"], donnees["y_reussite"]
        else:
            x, y = donnees["episodes"], donnees["couverture"]
        ax.plot(x, y, color=couleur, linewidth=1.5,
                marker="o" if est_reussite else None,
                markersize=2.5, label=f"epsilon = {epsilon:g}")

    ax.set_title(f"{titre} - {qualification}")
    ax.set_xlabel("Episode d'entrainement")
    if est_reussite:
        ax.set_ylabel(f"Succes sur chaque bloc de {bloc} episodes (%)")
    else:
        ax.set_ylabel("Couples (etat, action) essayes depuis le debut (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    ax.legend()
    dossier.mkdir(parents=True, exist_ok=True)
    nom = ("reussite" if est_reussite else "couverture")
    prefixe = {
        ("aleatoire", "reussite"): "1",
        ("aleatoire", "couverture"): "2",
        ("fixe", "reussite"): "3",
        ("fixe", "couverture"): "4",
    }[(depart, nom)]
    chemin = dossier / f"{prefixe}_{nom}_{depart}.png"
    fig.savefig(chemin, dpi=160)
    plt.close(fig)
    return chemin


def exporter_mesures(resultats, dossier, depart):
    chemin = dossier / f"mesures_{depart}.csv"
    with chemin.open("w", newline="", encoding="utf-8") as fichier:
        ecrivain = csv.writer(fichier)
        ecrivain.writerow((
            "epsilon", "episode", "reussite_brute_0_ou_1",
            "couverture_cumulative_pourcent"
        ))
        for epsilon, donnees in resultats.items():
            ecrivain.writerows(zip(
                [epsilon] * len(donnees["episodes"]),
                donnees["episodes"],
                donnees["reussites_brutes"],
                donnees["couverture"],
            ))
    return chemin


def comparer(epsilons=EPSILONS_PAR_DEFAUT, mode="distance", actions=9,
            episodes=5000, bloc=100, seed=42, angles_fixes=(90, 0),
            max_pas=50):
    if not epsilons or len(set(epsilons)) != len(epsilons):
        raise ValueError("Les valeurs d'epsilon doivent etre distinctes.")
    dossier = (DOSSIER_RESULTATS / "bras_robotique"
               / "comparaison_epsilon_reussite_couverture"
               / f"{actions}_actions" / mode)
    dossier.mkdir(parents=True, exist_ok=True)
    figures = []

    for depart in ("aleatoire", "fixe"):
        resultats = {}
        for epsilon in epsilons:
            resultats[epsilon] = entrainer_une_configuration(
                epsilon=float(epsilon),
                depart_aleatoire=(depart == "aleatoire"),
                mode=mode, actions=actions, angles_fixes=angles_fixes,
                episodes=episodes, bloc=bloc, seed=seed, max_pas=max_pas,
            )
        exporter_mesures(resultats, dossier, depart)
        for mesure in ("reussite", "couverture"):
            figures.append(tracer(
                resultats, mesure, depart, dossier, bloc, angles_fixes
            ))

    print(f"4 graphiques et 2 CSV dans : {dossier}")
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epsilons", nargs="+", type=float,
                        default=list(EPSILONS_PAR_DEFAUT))
    parser.add_argument("--mode", choices=("distance", "smart", "naive"),
                        default="distance")
    parser.add_argument("--actions", type=int, choices=(4, 9), default=9)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--bloc", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pas", type=int, default=50)
    parser.add_argument("--angles-fixes", nargs=2, type=int, default=[90, 0],
                        metavar=("THETA1", "THETA2"))
    args = parser.parse_args()
    comparer(epsilons=tuple(args.epsilons), mode=args.mode,
            actions=args.actions, episodes=args.episodes, bloc=args.bloc,
            seed=args.seed, angles_fixes=tuple(args.angles_fixes),
            max_pas=args.max_pas)


if __name__ == "__main__":
    main()
