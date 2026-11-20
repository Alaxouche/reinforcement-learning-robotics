"""Entraîner un réseau Keras : état s -> vecteur one-hot de l'action.

Lancement depuis la racine du dépôt :
    py -m experiences.reseau_bras --angles 180 90
    py -m experiences.reseau_bras --angles 180 90 --animer
    py -m experiences.reseau_bras --sauvegarder experiences/modeles_reseau/bras_9.keras
"""

import argparse

from experiences.animation_bras import animer, position_depart, trajectoire_q
from experiences.suivi_apprentissage import evaluer_globalement, preparer_transitions
from src.algorithmes.reseau_neurones import PolitiqueNeurones, distance_minimale
from src.environnements.bras_robotique.bras_robot_distance import BrasRobotDistance


def main():
    parser = argparse.ArgumentParser(
        description="Réseau Keras supervisé : un état s -> une action one-hot."
    )
    parser.add_argument("--actions", type=int, choices=(4, 9), default=9)
    parser.add_argument("--epoques", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--etat", type=int, help="État entier (de 0 à 1295).")
    parser.add_argument("--angles", nargs=2, type=int, metavar=("THETA1", "THETA2"))
    parser.add_argument("--charger", help="Recharger un modèle Keras (.keras).")
    parser.add_argument("--sauvegarder", help="Sauvegarder le modèle (.keras).")
    parser.add_argument("--animer", action="store_true")
    parser.add_argument("--gif", help="Exporter la trajectoire sous forme de GIF.")
    parser.add_argument("--sans-fenetre", action="store_true")
    args = parser.parse_args()

    if args.etat is not None and args.angles:
        parser.error("Choisir --etat OU --angles.")
    if args.charger and args.sauvegarder:
        parser.error("--charger et --sauvegarder sont incompatibles.")
    if args.sans_fenetre and not args.gif:
        parser.error("--sans-fenetre nécessite --gif.")
    if args.epoques <= 0:
        parser.error("--epoques doit être positif.")

    env = BrasRobotDistance(nb_actions=args.actions)
    agent = PolitiqueNeurones(env, seed=args.seed)

    if args.charger:
        agent.charger(args.charger)
        print(f"Modèle Keras chargé : {args.charger}")
    else:
        bonnes, total = agent.entrainer(epoques=args.epoques)
        print(f"Précision sur les données d'entraînement : {bonnes}/{total} ")
        if args.sauvegarder:
            agent.sauvegarder(args.sauvegarder)
            print(f"Modèle enregistré : {args.sauvegarder}")

    if args.etat is not None:
        if not 0 <= args.etat < env.nb_etats:
            parser.error(f"--etat doit être entre 0 et {env.nb_etats - 1}.")
        depart = args.etat
    else:
        try:
            depart = position_depart(
                env, angles=tuple(args.angles) if args.angles else None
            )
        except ValueError as erreur:
            parser.error(str(erreur))

    if depart in env.etats_objectif:
        parser.error("Ce départ est déjà sur la cible : aucune action.")

    vecteur = agent.predire(depart)
    indice = vecteur.index(1)
    print(f"\nÉtat s = {depart} ; angles = {env.angles(depart)}")
    print(f"Vecteur prédit = {vecteur}")
    print(f"Action i = {indice} : {env.noms_actions[indice]}")

    # Mesures de la politique Keras sur tous les départs, sans Q-table.
    transitions = preparer_transitions(env)
    reussis, total, taux, cout = evaluer_globalement(agent, transitions)
    cout_optimal = sum(distance_minimale(env, s) for s in env.etats_depart) / total
    print(f"\nRéussite : {reussis}/{total} ({taux:.2f} %)")
    print(f"Coût moyen du réseau : {cout:.3f} (échec pénalisé à 51)")
    print(f"Minimum théorique moyen : {cout_optimal:.3f}")

    if args.animer or args.gif:
        etats, actions, reussi, raison = trajectoire_q(agent, depart)
        print(f"Trajet du réseau : {len(actions)} mouvements, {raison}")
        animer(agent, etats, actions, reussi, raison,
               gif=args.gif, afficher=not args.sans_fenetre)


if __name__ == "__main__":
    main()
