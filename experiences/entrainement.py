"""Expérience : Q-learning sur le labyrinthe 5 x 5."""

from src.environnements.labyrinthe_5x5 import Labyrinthe5x5
from src.algorithmes.q_learning import QLearning


def main():
    """Crée le labyrinthe, entraîne le Q-learning et affiche la table Q."""

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

    return algorithme


if __name__ == "__main__":
    main()
