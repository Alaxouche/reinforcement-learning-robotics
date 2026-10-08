"""Expérience : Q-learning sur un labyrinthe simple."""

from src.environnements.labyrinthe_simple import LabyrintheSimple
from src.algorithmes.tabulaire.q_learning import QLearning


def main():
    """Crée l'environnement puis entraîne le Q-learning."""

    environnement = LabyrintheSimple(taille=3)

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
