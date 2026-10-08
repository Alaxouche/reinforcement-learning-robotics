"""Point de départ de la simulation : construit les deux objets puis ouvre Tkinter.

Commande depuis la racine : python -m experiences.q_learning_tabulaire.simulation
"""

from src.environnements.labyrinthe import creer_labyrinthe
from src.algorithmes.tabulaire.q_learning import QLearning
from src.visualisation.simulation_labyrinthe import lancer_simulation


CONFIGURATION_LABYRINTHE = "9x9"


def main():
    """Prépare l'environnement et l'agent puis lance la fenêtre. Retour : aucun."""
    # Changer uniquement cette valeur pour tester une autre configuration.
    lab = creer_labyrinthe(CONFIGURATION_LABYRINTHE)

    # Les hyperparamètres de cette expérience sont choisis ici.
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

    # On donne les DEUX objets à la visualisation : lab pour dessiner,
    # agent pour apprendre puis choisir ses déplacements.
    lancer_simulation(lab, agent)


if __name__ == "__main__":
    main()
