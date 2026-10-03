import random


class Labyrinthe:
    """Labyrinthe simple : une grille carree, sans mur ni feu.

    C'est le modele a suivre pour les autres labyrinthes. La classe QLearning n'utilise
    que ce qui est defini ici :
      - les attributs taille, nb_etats, depart, arrivee, actions, noms_actions
      - les methodes etat_suivant(etat, action) et recompense(etat)
    La simulation utilise en plus les listes murs et feux pour l'affichage.

    Pour un nouveau labyrinthe, on cree une classe fille qui remplit murs
    et/ou feux et qui redefinit etat_suivant et/ou recompense.
    """

    def __init__(self, taille=3):
        self.taille = taille
        self.nb_etats = taille * taille
        self.depart = 0
        self.arrivee = self.nb_etats - 1

        # Haut, Bas, Gauche, Droite : (deplacement en ligne, deplacement en colonne)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.noms_actions = ["Haut", "Bas", "Gauche", "Droite"]

        # Numeros des cases speciales (vides ici, a remplir dans les classes filles)
        self.murs = []
        self.feux = []

    def etat_suivant(self, etat, action):
        ligne, colonne = divmod(etat, self.taille)
        dl, dc = self.actions[action]
        ligne, colonne = ligne + dl, colonne + dc

        if 0 <= ligne < self.taille and 0 <= colonne < self.taille:
            return ligne * self.taille + colonne
        return etat  # contre un bord : l'agent reste sur place

    def recompense(self, etat):
        return 1.0 if etat == self.arrivee else -0.1



class LabyrintheAleatoire(Labyrinthe):
    """Labyrinthe avec des murs et des feux places au hasard.

    Un mur bloque l'agent comme un bord. Un feu donne une penalite de -10.
    Le tirage garantit qu'un chemin sans feu mene du depart a l'arrivee.
    Une meme graine redonne toujours le meme labyrinthe.
    """

    def __init__(self, taille, nb_murs, nb_feux=0, graine=None):
        Labyrinthe.__init__(self, taille)
        rng = random.Random(graine)
        cases_libres = [c for c in range(self.nb_etats) if c not in (self.depart, self.arrivee)]

        while True:
            tirage = rng.sample(cases_libres, nb_murs + nb_feux)
            self.murs = tirage[:nb_murs]
            self.feux = tirage[nb_murs:]
            if self.arrivee_accessible():
                break

    def etat_suivant(self, etat, action):
        suivant = Labyrinthe.etat_suivant(self, etat, action)
        return etat if suivant in self.murs else suivant  # un mur bloque comme un bord

    def recompense(self, etat):
        return -10.0 if etat in self.feux else Labyrinthe.recompense(self, etat)

    def arrivee_accessible(self):
        a_visiter, vus = [self.depart], {self.depart}
        while a_visiter:
            etat = a_visiter.pop()
            if etat == self.arrivee:
                return True
            for action in range(len(self.actions)):
                suivant = self.etat_suivant(etat, action)
                if suivant not in vus and suivant not in self.feux:
                    vus.add(suivant)
                    a_visiter.append(suivant)
        return False


# Labyrinthes predefinis : creer_labyrinthe("murs_5x5") ou creer_labyrinthe("murs_5x5", graine=42)
LABYRINTHES = {
    "simple_3x3": lambda graine: Labyrinthe(3),
    "murs_5x5": lambda graine: LabyrintheAleatoire(5, nb_murs=6, graine=graine),
    "murs_feux_9x9": lambda graine: LabyrintheAleatoire(9, nb_murs=27, nb_feux=5, graine=graine),
}


def creer_labyrinthe(nom, graine=None):
    return LABYRINTHES[nom](graine)
