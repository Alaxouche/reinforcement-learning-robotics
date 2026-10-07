"""Création d'un environnement puis entraînement d'un agent Q-learning."""

from src.environnements.labyrinthe import Labyrinthe
from src.algorithmes.tabulaire.q_learning import QLearning


def main():
    """Construit le labyrinthe, l'agent, puis lance l'apprentissage."""

    # Environnement concret : Labyrinthe hérite de Environnement.
    lab = Labyrinthe(taille=3)

    # Les hyperparamètres appartiennent à l'expérience, pas à l'environnement.
    agent = QLearning(
        environnement=lab,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.05,
        decroissance=0.99,
        nb_episodes=1000,
        max_pas=30
    )

    agent.apprendre()

    return agent


if __name__ == "__main__":
    main()
