"""Expérience concrète : Q-learning tabulaire sur un labyrinthe."""

from src.environnements.labyrinthe import Labyrinthe
from src.algorithmes.tabulaire.q_learning import QLearning


def main():
    """Assemble un environnement concret et un algorithme concret."""

    # L'expérience choisit ici UN environnement parmi tous ceux disponibles.
    environnement = Labyrinthe(taille=3)

    # L'expérience choisit ici UN algorithme parmi tous ceux disponibles.
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

    return algorithme


if __name__ == "__main__":
    main()
