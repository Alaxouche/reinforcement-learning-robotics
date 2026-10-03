import numpy as np

from labyrinthe import creer_labyrinthe


class QLearning:
    """Apprentissage par Q-learning qui fonctionne avec n'importe quel
    labyrinthe respectant le modele de la classe Labyrinthe."""

    def __init__(self, labyrinthe, alpha=0.1, gamma=0.9, epsilon_debut=1.0, epsilon_min=0.05, decroissance=0.99, nb_episodes=500, max_pas=50, graine=0):
        self.lab = labyrinthe
        self.alpha = alpha                  # vitesse d'apprentissage
        self.gamma = gamma                  # importance du futur
        self.epsilon_debut = epsilon_debut  # exploration totale au debut
        self.epsilon_min = epsilon_min      # exploration minimale
        self.decroissance = decroissance    # epsilon est multiplie par ce nombre a chaque episode
        self.nb_episodes = nb_episodes
        self.max_pas = max_pas
        self.rng = np.random.default_rng(graine)  # meme graine = memes resultats

        self.Q = np.zeros((labyrinthe.nb_etats, len(labyrinthe.actions)))

    def meilleure_action(self, etat):
        # En cas d'egalite, on tire au hasard parmi les meilleures actions
        meilleures = np.flatnonzero(self.Q[etat] == self.Q[etat].max())
        return int(self.rng.choice(meilleures))

    def choisir_action(self, etat, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(len(self.lab.actions)))
        return self.meilleure_action(etat)

    def apprendre(self):
        epsilon = self.epsilon_debut

        for episode in range(self.nb_episodes):
            etat = self.lab.depart

            for pas in range(self.max_pas):
                action = self.choisir_action(etat, epsilon)
                suivant = self.lab.etat_suivant(etat, action)
                r = self.lab.recompense(suivant)

                if suivant == self.lab.arrivee:
                    cible = r  # etat final : pas de futur
                else:
                    cible = r + self.gamma * self.Q[suivant].max()
                self.Q[etat, action] += self.alpha * (cible - self.Q[etat, action])

                etat = suivant
                if etat == self.lab.arrivee:
                    break

            epsilon = max(self.epsilon_min, epsilon * self.decroissance)

    def chemin(self):
        etat = self.lab.depart
        parcours = [etat]
        while etat != self.lab.arrivee and len(parcours) <= self.max_pas:
            etat = self.lab.etat_suivant(etat, self.meilleure_action(etat))
            parcours.append(etat)
        return parcours

    def afficher_q(self):
        entete = "".join("{:>10}".format(nom) for nom in self.lab.noms_actions)
        print("etat{}   meilleure".format(entete))
        for etat in range(self.lab.nb_etats):
            if etat == self.lab.arrivee:
                meilleure = "arrivee"
            else:
                meilleure = self.lab.noms_actions[int(np.argmax(self.Q[etat]))]
            valeurs = "".join("{:10.3f}".format(v) for v in self.Q[etat])
            print("{:4d}{}   {}".format(etat, valeurs, meilleure))


if __name__ == "__main__":
    # Pour changer de labyrinthe, il suffit de changer ces deux lignes.
    # Exemple pour un grand labyrinthe : QLearning(lab, nb_episodes=2000, max_pas=200)
    lab = creer_labyrinthe("simple_3x3")  # ou "murs_5x5", "murs_feux_9x9" 
    qlearning = QLearning(lab)

    qlearning.apprendre()
    qlearning.afficher_q()
    print()
    print("chemin :", qlearning.chemin())
