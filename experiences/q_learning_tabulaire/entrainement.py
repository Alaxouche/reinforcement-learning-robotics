"""Expérience : Q-learning sur le labyrinthe 4 x 4."""

from src.environnements.labyrinthe_4x4 import Labyrinthe4x4
from src.algorithmes.tabulaire.q_learning import QLearning


def main():
    """Crée le labyrinthe puis entraîne le Q-learning."""

    environnement = Labyrinthe4x4()

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
