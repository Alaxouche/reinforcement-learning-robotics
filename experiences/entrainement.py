"""Expérience : Q-learning sur le labyrinthe 5 x 5."""

from src.environnements.labyrinthe_5x5 import Labyrinthe5x5
from src.algorithmes.q_learning import QLearning


def main():
    """Crée le labyrinthe, entraîne le Q-learning et affiche les résultats."""

    environnement = Labyrinthe5x5()

    algorithme = QLearning(
        environnement=environnement,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.05,
        decroissance=0.99,
        nb_episodes=1000,
        max_pas=30
    )

    algorithme.apprendre()

    # Affichage propre de la table Q.
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

    # Détection du chemin appris par l'agent.
    etat = environnement.reset()
    chemin = [etat]
    visites = {etat}
    objectif_atteint = False
    boucle_detectee = False

    for _ in range(algorithme.max_pas):
        if etat == environnement.arrivee:
            objectif_atteint = True
            break

        actions = environnement.actions_possibles(etat)

        if not actions:
            break

        # Après l'entraînement, on n'explore plus :
        # on choisit toujours l'action ayant la plus grande Q-value.
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

    print("\nCHEMIN OPTIMAL DÉTECTÉ PAR L'AGENT")
    print("=" * 62)
    print(" -> ".join(str(etat) for etat in chemin))
    print(f"Nombre de déplacements : {len(chemin) - 1}")

    if objectif_atteint:
        print("Objectif atteint : Oui")
    elif boucle_detectee:
        print("Objectif atteint : Non (boucle détectée)")
    else:
        print("Objectif atteint : Non")

    print("=" * 62)

    return algorithme


if __name__ == "__main__":
    main()
