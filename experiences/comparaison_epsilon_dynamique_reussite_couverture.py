"""Quatre graphes : reussite et couverture avec epsilon decroissant.

Deux paquets (departs aleatoires / depart fixe), trois profils par figure :
rapide, moyenne et lente. Parametres repris de etude_epsilon_depart.py.

Reussite = nombre exact de succes dans chaque bloc NON CHEVAUCHANT divise
par la taille du bloc. Ce n'est pas une moyenne glissante.
Couverture = proportion cumulative exacte de couples (etat, action) essayes,
mesuree a la fin de CHAQUE episode.
Les valeurs brutes par episode sont exportees en CSV.

Lancer depuis la racine du projet :
    py -m experiences.comparaison_epsilon_dynamique_reussite_couverture
    py -m experiences.comparaison_epsilon_dynamique_reussite_couverture --angles-fixes 180 90
"""

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiences.entrainement_bras_robotique import MODES_BRAS
from experiences.etude_epsilon_depart import (
    DECROISSANCES, EPSILON_INITIAL, EPSILON_MIN,
)
from experiences.graphes import DOSSIER_RESULTATS, config_bras, creer_agent


PROFILS = ("rapide", "moyenne", "lente")
COULEURS = {"rapide": "#C9363C", "moyenne": "#1764B0", "lente": "#16865D"}
ETIQUETTES = {
    "rapide": "Rapide (lambda = 0.99)",
    "moyenne": "Normale (lambda = 0.997)",
    "lente": "Lente (lambda = 0.9995)",
}


def entrainer_profil(profil, depart_aleatoire, mode="distance", actions=9,
                    angles_fixes=(90, 0), episodes=5000, bloc=100,
                    seed=42, max_pas=50):
    """Mesure le succes reel de chaque episode, avec une Q-table initiale nulle."""
    if profil not in DECROISSANCES:
        raise ValueError(f"Profil inconnu : {profil}")
    if episodes <= 0 or bloc <= 0 or max_pas <= 0:
        raise ValueError("episodes, bloc et max_pas doivent etre positifs.")

    config = config_bras(
        mode, actions, depart_aleatoire=depart_aleatoire,
        epsilon=EPSILON_INITIAL, epsilon_min=EPSILON_MIN,
        decroissance=DECROISSANCES[profil],
        nb_episodes=1, max_pas=max_pas,
    )
    if not depart_aleatoire:
        # reset() revient sur la meme pose au debut de TOUS les episodes.
        config["construire"] = lambda graine: MODES_BRAS[mode](
            depart_aleatoire=False, angles_initiaux=tuple(angles_fixes),
            seed=graine, nb_actions=actions,
        )

    agent, env = creer_agent(config, seed)
    assert not np.count_nonzero(agent.Q), "La Q-table initiale doit etre nulle."
    assert env.depart_aleatoire == depart_aleatoire
    if not depart_aleatoire:
        assert env.angles(env.etat_initial) == tuple(angles_fixes)
        if env.etat_initial not in env.etats_depart:
            raise ValueError("Le depart fixe est deja sur la cible.")

    etats_non_terminaux = np.asarray(env.etats_depart, dtype=int)
    nb_couples = len(etats_non_terminaux) * env.nb_actions
    reussite_brute = np.empty(episodes, dtype=int)
    couverture = np.empty(episodes, dtype=float)
    epsilon_utilise = np.empty(episodes, dtype=float)

    for i in range(episodes):
        epsilon_utilise[i] = agent.epsilon
        # apprendre() n'effectue ici qu'un episode. La Q-table et les compteurs
        # sont conserves entre les appels, seule la position est reinitialisee.
        agent.apprendre()
        reussite_brute[i] = int(env.distance_cible(env.etat) <= env.tolerance)
        couverture[i] = (
            100.0 * np.count_nonzero(agent.mises_a_jour[etats_non_terminaux])
            / nb_couples
        )

    x_bloc = []
    succes_bloc = []
    for debut in range(0, episodes, bloc):
        fin = min(debut + bloc, episodes)
        nb_reussis = int(reussite_brute[debut:fin].sum())
        x_bloc.append(fin)
        succes_bloc.append(100.0 * nb_reussis / (fin - debut))

    return {
        "episode": np.arange(1, episodes + 1, dtype=int),
        "reussite_brute": reussite_brute,
        "couverture": couverture,
        "epsilon_utilise": epsilon_utilise,
        "x_bloc": np.asarray(x_bloc, dtype=int),
        "succes_bloc": np.asarray(succes_bloc, dtype=float),
    }


def tracer(resultats, type_mesure, depart, dossier, bloc, angles_fixes):
    """Une courbe par profil, sans interpolation statistique ni lissage."""
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    for profil, donnees in resultats.items():
        if type_mesure == "reussite":
            x = donnees["x_bloc"]
            y = donnees["succes_bloc"]
        elif type_mesure == "couverture":
            x = donnees["episode"]
            y = donnees["couverture"]
        else:
            raise ValueError("Mesure inconnue.")
        ax.plot(
            x, y, color=COULEURS[profil],
            linewidth=1.5, label=ETIQUETTES[profil],
            marker="o" if type_mesure == "reussite" else None,
            markersize=2.5,
        )

    titre = ("Reussite" if type_mesure == "reussite"
             else "Couverture des couples (etat, action)")
    depart_titre = ("departs aleatoires" if depart == "aleatoire"
                    else f"depart fixe {tuple(angles_fixes)}")
    ax.set(
        title=f"{titre} - epsilon dynamique - {depart_titre}",
        xlabel="Episode d'entrainement",
        ylabel=(f"Succes par bloc de {bloc} episodes (%)"
                if type_mesure == "reussite"
                else "Couples (etat, action) essayes (%)"),
    )
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    ax.legend()
    prefixe = {
        ("aleatoire", "reussite"): "1",
        ("aleatoire", "couverture"): "2",
        ("fixe", "reussite"): "3",
        ("fixe", "couverture"): "4",
    }[(depart, type_mesure)]
    chemin = dossier / f"{prefixe}_{type_mesure}_{depart}.png"
    fig.savefig(chemin, dpi=160)
    plt.close(fig)
    return chemin


def exporter_brut(resultats, dossier, depart):
    """Un enregistrement EXACT par episode et par profil."""
    chemin = dossier / f"mesures_{depart}.csv"
    with chemin.open("w", newline="", encoding="utf-8") as fichier:
        writer = csv.writer(fichier)
        writer.writerow((
            "profil", "lambda", "episode", "epsilon_utilise",
            "reussite_brute_0_ou_1", "couverture_cumulative_pourcent",
        ))
        for profil, donnees in resultats.items():
            writer.writerows(zip(
                [profil] * len(donnees["episode"]),
                [DECROISSANCES[profil]] * len(donnees["episode"]),
                donnees["episode"],
                donnees["epsilon_utilise"],
                donnees["reussite_brute"],
                donnees["couverture"],
            ))
    return chemin


def comparer(profils=PROFILS, mode="distance", actions=9, episodes=5000,
            bloc=100, seed=42, angles_fixes=(90, 0), max_pas=50):
    if not profils or any(p not in PROFILS for p in profils):
        raise ValueError("Profils attendus : rapide, moyenne, lente.")
    if len(set(profils)) != len(profils):
        raise ValueError("Les profils doivent etre distincts.")
    dossier = (DOSSIER_RESULTATS / "bras_robotique"
               / "comparaison_epsilon_dynamique_reussite_couverture"
               / f"{actions}_actions" / mode)
    dossier.mkdir(parents=True, exist_ok=True)

    chemins = []
    for depart in ("aleatoire", "fixe"):
        resultat_par_profil = {}
        for profil in profils:
            resultat_par_profil[profil] = entrainer_profil(
                profil, depart_aleatoire=(depart == "aleatoire"),
                mode=mode, actions=actions,
                angles_fixes=angles_fixes, episodes=episodes, bloc=bloc,
                seed=seed, max_pas=max_pas,
            )
        exporter_brut(resultat_par_profil, dossier, depart)
        for mesure in ("reussite", "couverture"):
            chemins.append(tracer(
                resultat_par_profil, mesure, depart, dossier, bloc,
                angles_fixes,
            ))

    print(f"4 graphes et 2 CSV enregistres dans : {dossier}")
    return chemins


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profils", nargs="+", choices=PROFILS,
                        default=list(PROFILS))
    parser.add_argument("--mode", choices=("distance", "smart", "naive"),
                        default="distance")
    parser.add_argument("--actions", type=int, choices=(4, 9), default=9)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--bloc", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pas", type=int, default=50)
    parser.add_argument("--angles-fixes", nargs=2, type=int,
                        default=[90, 0], metavar=("THETA1", "THETA2"))
    args = parser.parse_args()
    comparer(
        profils=tuple(args.profils), mode=args.mode,
        actions=args.actions, episodes=args.episodes, bloc=args.bloc,
        seed=args.seed, angles_fixes=tuple(args.angles_fixes),
        max_pas=args.max_pas,
    )


if __name__ == "__main__":
    main()
