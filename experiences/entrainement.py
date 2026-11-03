"""Expérience : Q-learning sur les labyrinthes 5 x 5 et 36 x 36.

Lancement depuis la racine du projet :
  py -m experiences.entrainement              # les deux labyrinthes
  py -m experiences.entrainement --taille 5   # un seul
"""

import argparse
import time

import numpy as np

from src.environnements.labyrinthe.labyrinthe_5x5 import Labyrinthe5x5
from src.environnements.labyrinthe.labyrinthe_36x36 import Labyrinthe36x36
from src.algorithmes.q_learning import QLearning


# Le 36 x 36 garde les récompenses du 5 x 5. Grille ouverte (15 % de murs,
# 60 pièges) : beaucoup de chemins possibles, plus court chemin de 70 pas
# (8 pour le 5 x 5). Il faut un horizon plus long (gamma), des épisodes plus
# longs (max_pas), plus d'épisodes et un alpha plus élevé (0.1 ne suffit pas).
CONFIGURATIONS = {
    5: {"classe": Labyrinthe5x5, "alpha": 0.1, "gamma": 0.9, "epsilon_min": 0.05,
        "decroissance": 0.99, "nb_episodes": 1000, "max_pas": 30},
    36: {"classe": Labyrinthe36x36, "alpha": 0.5, "gamma": 0.99, "epsilon_min": 0.05,
         "decroissance": 0.998, "nb_episodes": 3000, "max_pas": 1000},
}


def afficher_table_q(algorithme, environnement):
    print("\nTABLE Q")
    print("=" * 62)
    entete = f"{'Etat':>6}"
    for nom_action in environnement.noms_actions:
        entete += f"{nom_action:>14}"
    print(entete)
    print("-" * 62)
    for etat in range(environnement.nb_etats):
        ligne = f"{etat:>6}"
        for action in range(environnement.nb_actions):
            ligne += f"{algorithme.Q[etat, action]:>14.3f}"
        print(ligne)
    print("=" * 62)


def suivre_chemin(algorithme, environnement):
    """Suit la politique gloutonne depuis le départ ; s'arrête sur une boucle."""
    etat = environnement.reset()
    chemin = [etat]
    visites = {etat}
    objectif_atteint = False
    boucle_detectee = False
    for _ in range(algorithme.max_pas):
        if not environnement.actions_possibles(etat):
            break
        action = algorithme.meilleure_action(etat)
        nouvel_etat, _, termine = environnement.step(action)
        chemin.append(nouvel_etat)
        if termine:
            objectif_atteint = True
            break
        if nouvel_etat in visites:
            boucle_detectee = True
            break
        visites.add(nouvel_etat)
        etat = nouvel_etat
    return chemin, objectif_atteint, boucle_detectee


def entrainer_labyrinthe(taille, seed=42):
    config = CONFIGURATIONS[taille]
    np.random.seed(seed)
    environnement = config["classe"]()
    algorithme = QLearning(
        environnement=environnement,
        alpha=config["alpha"], gamma=config["gamma"], epsilon=1.0,
        epsilon_min=config["epsilon_min"], decroissance=config["decroissance"],
        nb_episodes=config["nb_episodes"], max_pas=config["max_pas"],
    )
    debut = time.perf_counter()
    algorithme.apprendre()
    duree = time.perf_counter() - debut
    return algorithme, environnement, duree


def main(tailles=(5, 36)):
    resultats = {}
    for taille in tailles:
        algorithme, environnement, duree = entrainer_labyrinthe(taille)
        print(f"\n######## LABYRINTHE {taille}x{taille} ########")

        # 1 296 lignes pour le 36 x 36 : la table n'est affichée que pour le 5 x 5.
        if environnement.nb_etats <= 100:
            afficher_table_q(algorithme, environnement)

        chemin, objectif_atteint, boucle_detectee = suivre_chemin(algorithme, environnement)
        print("\nCHEMIN APPRIS PAR L'AGENT")
        print("=" * 62)
        if len(chemin) <= 40:
            print(" -> ".join(str(etat) for etat in chemin))
        else:
            print(" -> ".join(str(etat) for etat in chemin[:10]) + " -> ... -> "
                  + " -> ".join(str(etat) for etat in chemin[-5:]))
        print(f"Nombre de déplacements : {len(chemin) - 1}")
        if objectif_atteint:
            print("Objectif atteint : Oui")
        elif boucle_detectee:
            print("Objectif atteint : Non (boucle détectée)")
        else:
            print("Objectif atteint : Non")
        print(f"Durée d'entraînement : {duree:.1f} s")
        print("=" * 62)
        resultats[taille] = (algorithme, len(chemin) - 1, objectif_atteint, duree)

    if len(resultats) > 1:
        print("\nCOMPARAISON")
        print(f"{'':<34}" + "".join(f"{f'{t}x{t}':>12}" for t in resultats))
        lignes = [
            ("états", lambda a, *_: a.env.nb_etats),
            ("murs", lambda a, *_: len(a.env.murs)),
            ("épisodes", lambda a, *_: a.nb_episodes),
            ("pas moyens (10 derniers %)",
             lambda a, *_: f"{np.mean(a.nombre_actions[-a.nb_episodes // 10:]):.1f}"),
            ("pas totaux d'entraînement", lambda a, *_: sum(a.nombre_actions)),
            ("déplacements du chemin appris", lambda a, n, *_: n),
            ("objectif atteint", lambda a, n, ok, _: "Oui" if ok else "Non"),
            ("durée (s)", lambda a, n, ok, d: f"{d:.1f}"),
        ]
        for titre, valeur in lignes:
            print(f"{titre:<34}" + "".join(f"{str(valeur(*r)):>12}" for r in resultats.values()))
    return resultats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--taille", type=int, choices=sorted(CONFIGURATIONS),
                        action="append", help="5 ou 36 (par défaut : les deux)")
    arguments = parser.parse_args()
    main(tuple(arguments.taille) if arguments.taille else (5, 36))
