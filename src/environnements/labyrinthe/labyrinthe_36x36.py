import random
from collections import deque

from src.environnements.labyrinthe.labyrinthe import Labyrinthe


class Labyrinthe36x36(Labyrinthe):
    """Grand labyrinthe 36 x 36 (1 296 états), ouvert, généré de façon reproductible.

    Contrairement à un labyrinthe « parfait » (un seul couloir vers la sortie),
    la grille est largement ouverte : des segments de murs droits sont posés au
    hasard (environ 15 % des cases), si bien que de très nombreux chemins mènent
    à l'arrivée. Des pièges sont dispersés sur la grille : le plus court chemin
    en nombre de pas n'est donc pas forcément le meilleur, et l'agent doit
    arbitrer entre longueur du trajet et pièges traversés.

    Départ : 0 (coin haut gauche)
    Arrivée : 1295 (coin bas droit)

    Les récompenses sont identiques à celles du 5 x 5 :
    -0.1 par déplacement, -1.0 sur un piège, +1.0 à l'arrivée.
    Un piège coûte donc autant que 10 déplacements : un détour de moins de
    10 pas pour l'éviter est rentable.
    """

    TAILLE = 36

    def __init__(self, seed=0, densite_murs=0.15, nb_pieges=60):
        taille = self.TAILLE
        arrivee = taille * taille - 1
        generateur = random.Random(seed)  # indépendant de np.random : Q identique

        murs = self._poser_murs(taille, generateur, densite_murs, depart=0, arrivee=arrivee)
        super().__init__(taille=taille, depart=0, arrivee=arrivee, murs=sorted(murs))

        proteges = self._voisinage(0, 2) | self._voisinage(arrivee, 2)
        libres = set(range(self.nb_etats)) - set(self.murs)
        self.pieges = sorted(generateur.sample(sorted(libres - proteges), nb_pieges))
        self._ensemble_pieges = set(self.pieges)

    @classmethod
    def _poser_murs(cls, taille, generateur, densite_murs, depart, arrivee):
        """Pose des segments de murs ; les poches isolées sont ensuite comblées."""
        def voisinage(etat):
            ligne, colonne = divmod(etat, taille)
            return {l * taille + c for l in range(ligne - 1, ligne + 2)
                    for c in range(colonne - 1, colonne + 2)
                    if 0 <= l < taille and 0 <= c < taille}

        interdits = voisinage(depart) | voisinage(arrivee)
        murs = set()
        while len(murs) < densite_murs * taille * taille:
            ligne = generateur.randrange(taille)
            colonne = generateur.randrange(taille)
            longueur = generateur.randint(3, 8)
            dl, dc = generateur.choice(((0, 1), (1, 0)))
            segment = [(ligne + i * dl) * taille + (colonne + i * dc)
                       for i in range(longueur)
                       if ligne + i * dl < taille and colonne + i * dc < taille]
            if not interdits.intersection(segment):
                murs.update(segment)

        # Toute case libre non reliée au départ devient un mur : pas de poche inutile.
        atteintes = {depart}
        file = deque([depart])
        while file:
            ligne, colonne = divmod(file.popleft(), taille)
            for dl, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                l, c = ligne + dl, colonne + dc
                suivant = l * taille + c
                if (0 <= l < taille and 0 <= c < taille
                        and suivant not in murs and suivant not in atteintes):
                    atteintes.add(suivant)
                    file.append(suivant)
        if arrivee not in atteintes:
            raise ValueError("Arrivée inatteignable : changer la graine ou la densité.")
        return set(range(taille * taille)) - atteintes

    def _voisinage(self, etat, rayon):
        ligne, colonne = divmod(etat, self.taille)
        return {l * self.taille + c
                for l in range(ligne - rayon, ligne + rayon + 1)
                for c in range(colonne - rayon, colonne + rayon + 1)
                if 0 <= l < self.taille and 0 <= c < self.taille}

    def recompense(self, etat):
        """Renvoie la récompense, identique à celle du 5 x 5."""
        if etat == self.arrivee:
            return 1.0

        if etat in self._ensemble_pieges:
            return -1.0

        return -0.1
