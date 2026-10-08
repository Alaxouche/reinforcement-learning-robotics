"""Point de départ de l'expérience : création du labyrinthe et de l'agent.

Commande depuis la racine : python -m experiences.q_learning_tabulaire.entrainement
"""

from src.environnements.labyrinthe import creer_labyrinthe
from src.algorithmes.tabulaire.q_learning import QLearning


CONFIGURATION_LABYRINTHE = "9x9"


def main():
    """Crée les deux objets, entraîne l'agent et affiche le résultat. Retour : aucun."""
    # Changer uniquement cette valeur pour tester une autre configuration.
    lab = creer_labyrinthe(CONFIGURATION_LABYRINTHE)

    # 2. On TRANSMET l'objet lab à QLearning : dans sa classe, self.env = lab.
    # Les hyperparamètres de cette expérience sont définis ici, pas dans l'algorithme.
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

    # 3. Cette méthode fait interagir agent et lab, puis remplit agent.Q.
    agent.apprendre()

    # 4. Affichage lisible de la table Q
    print("État    Haut      Bas    Gauche   Droite")
    # On parcourt une ligne de Q par état ; les 4 colonnes sont les actions.
    for etat in range(lab.nb_etats):
        valeurs = " ".join(f"{valeur:8.3f}" for valeur in agent.Q[etat])
        print(f"{etat:>4} {valeurs}")

    # 5. Test : l'agent suit uniquement les meilleures actions apprises.
    etat = lab.reset()  # Réinitialiser le labyrinthe APRÈS l'entraînement.
    chemin = [etat]
    score = 0.0
    issue = "trop de pas"

    for pas in range(agent.max_pas):
        action = agent.meilleure_action(etat)  # Plus d'exploration : on teste la politique apprise.
        etat, recompense, termine = lab.step(action)  # Le labyrinthe exécute le déplacement.
        chemin.append(etat)
        score += recompense

        if termine:
            # L'épisode peut se terminer de deux façons : l'arrivée ou un feu.
            issue = "arrivée" if etat == lab.arrivee else "feu"
            break

    print("\nChemin suivi :", " -> ".join(map(str, chemin)))
    print(f"Pas : {len(chemin) - 1}    Score : {score:.2f}    Issue : {issue}")


# Ce bloc lance main() seulement si l'on exécute ce fichier comme programme.
if __name__ == "__main__":
    main()
