"""Bras robotique : impact d'epsilon sur l'exploration, Q initiale constante ou aléatoire.

Pour isoler l'effet d'epsilon, il reste constant pendant tout l'entraînement
(epsilon_min = epsilon, décroissance = 1). Deux initialisations de la Q-table :

    constante : Q(s, a) = 0 partout (aucun tirage aléatoire) ;
    aleatoire : Q(s, a) tirée uniformément dans [-0.1, 0.1] (comme les autres expériences).

Tous les autres réglages sont ceux des expériences de référence du bras
(alpha = 0.2, gamma = 0.95, 5 000 épisodes, 50 mouvements maximum, départs
aléatoires). Mesures : part des couples (s, a) essayés au moins une fois,
réussite gloutonne sur 30 départs de test fixes, longueur des 500 derniers
épisodes. Les graphes sont écrits dans experiences/resultats/bras_robotique/epsilon/.

Lancement depuis la racine du projet (~3 minutes) :
  py -m experiences.etude_epsilon
  py -m experiences.etude_epsilon --actions 4 --modes naive smart --seeds 42 43 44
"""

import argparse

import numpy as np

from experiences.entrainement_bras_robotique import MODES_BRAS
from experiences.graphes import DOSSIER_RESULTATS, config_bras, entrainer_avec_suivi
from src.visualisation import graphes


EPSILONS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)
INITIALISATIONS = {
    "constante": {"initialisation": "constante", "valeur_initiale": 0.0},
    "aleatoire": {"initialisation": "aleatoire"},
}
PAS_SUIVI = 250          # épisodes entre deux mesures
FENETRE_FINALE = 500     # épisodes utilisés pour la longueur finale
NB_DEPARTS_TEST = 30


def config_epsilon(mode, epsilon, initialisation, nb_actions=9, nb_episodes=5000):
    """Configuration de référence du bras, avec un epsilon constant."""
    return config_bras(mode, nb_actions, epsilon=epsilon, epsilon_min=epsilon,
                       decroissance=1.0, nb_episodes=nb_episodes,
                       **INITIALISATIONS[initialisation])


def mesurer(mode, epsilon, initialisation, seed, nb_actions=9, nb_episodes=5000,
            pas_suivi=PAS_SUIVI, nb_tests=NB_DEPARTS_TEST):
    """Un entraînement ; renvoie les bilans successifs et les mesures finales."""
    config = config_epsilon(mode, epsilon, initialisation, nb_actions, nb_episodes)
    algorithme, environnement, evaluations = entrainer_avec_suivi(
        config, seed, pas_evaluation=pas_suivi, nb_tests=nb_tests
    )
    dernier = evaluations[-1]
    longueurs = algorithme.nombre_actions[-min(FENETRE_FINALE, len(algorithme.nombre_actions)):]
    mises_a_jour = algorithme.mises_a_jour.sum()
    final = {
        "couples_essayes": dernier["couples_essayes"],
        "taux_reussite": dernier["taux_reussite"],
        "longueur_finale": float(np.mean(longueurs)),
        "actions_aleatoires": 100.0 * algorithme.explorations.sum() / mises_a_jour,
        "etats_visites": 100.0 * np.count_nonzero(algorithme.visites) / environnement.nb_etats,
    }
    return {"evaluations": evaluations, "final": final}


def moyenne_seeds(resultats):
    """Moyenne, graine par graine, des bilans et des mesures finales."""
    if len(resultats) == 1:
        return resultats[0]
    evaluations = []
    for bilans in zip(*(r["evaluations"] for r in resultats)):
        evaluations.append({
            "episode": bilans[0]["episode"],
            **{cle: float(np.mean([b[cle] for b in bilans]))
               for cle in ("taux_reussite", "couples_essayes")},
        })
    final = {cle: float(np.mean([r["final"][cle] for r in resultats]))
             for cle in resultats[0]["final"]}
    return {"evaluations": evaluations, "final": final}


def etudier(modes=tuple(MODES_BRAS), epsilons=EPSILONS, seeds=(42,), nb_actions=9,
            nb_episodes=5000, pas_suivi=PAS_SUIVI, nb_tests=NB_DEPARTS_TEST):
    """resultats[mode][initialisation][epsilon] = {"evaluations", "final"}."""
    resultats = {}
    for mode in modes:
        resultats[mode] = {}
        for initialisation in INITIALISATIONS:
            resultats[mode][initialisation] = {}
            for epsilon in epsilons:
                resultats[mode][initialisation][epsilon] = moyenne_seeds([
                    mesurer(mode, epsilon, initialisation, seed, nb_actions,
                            nb_episodes, pas_suivi, nb_tests)
                    for seed in seeds
                ])
                final = resultats[mode][initialisation][epsilon]["final"]
                print(f"{mode:>8}  {initialisation:>9}  epsilon = {epsilon:<5g}"
                      f"couples essayés {final['couples_essayes']:5.1f} %  "
                      f"réussite {final['taux_reussite']:5.1f} %  "
                      f"mouvements/épisode {final['longueur_finale']:5.1f}  "
                      f"actions au hasard {final['actions_aleatoires']:5.1f} %")
    return resultats


def tracer(resultats, dossier, nb_actions):
    dossier.mkdir(parents=True, exist_ok=True)
    graphes.graphe_epsilon_evolution(
        resultats, "couples_essayes", "couples (s, a) essayés (%)",
        f"Bras robotique ({nb_actions} actions) : couverture des couples (s, a) "
        "selon epsilon", dossier / "couverture_couples.png",
    )
    graphes.graphe_epsilon_evolution(
        resultats, "taux_reussite", "réussite gloutonne (%)",
        f"Bras robotique ({nb_actions} actions) : réussite de la politique gloutonne "
        "selon epsilon", dossier / "reussite.png",
    )
    graphes.graphe_epsilon_bilan(resultats, dossier / "bilan_epsilon.png")


def main(modes=tuple(MODES_BRAS), seeds=(42,), nb_actions=9, nb_episodes=5000):
    print(f"Bras robotique, {nb_actions} actions, epsilon constant, "
          f"graine(s) {', '.join(map(str, seeds))}")
    resultats = etudier(modes=modes, seeds=seeds, nb_actions=nb_actions,
                        nb_episodes=nb_episodes)
    dossier = DOSSIER_RESULTATS / "bras_robotique" / "epsilon" / f"{nb_actions}_actions"
    tracer(resultats, dossier, nb_actions)
    print(f"\nGraphes écrits dans {dossier}")
    return resultats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Impact d'epsilon sur l'exploration du bras robotique."
    )
    parser.add_argument("--modes", nargs="+", choices=tuple(MODES_BRAS),
                        default=list(MODES_BRAS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--actions", type=int, choices=(4, 9), default=9)
    parser.add_argument("--episodes", type=int, default=5000)
    arguments = parser.parse_args()
    main(modes=tuple(arguments.modes), seeds=tuple(arguments.seeds),
         nb_actions=arguments.actions, nb_episodes=arguments.episodes)
