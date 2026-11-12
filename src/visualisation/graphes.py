"""Fonctions de tracé des graphes d'analyse du Q-learning (matplotlib).

Elles acceptent un labyrinthe comme le bras robotique : dans les deux cas, les
états forment une grille et etat = ligne * nb_colonnes + colonne. Pour le bras,
une ligne est un angle theta1 et une colonne un angle theta2.
"""

import heapq

import numpy as np
import matplotlib

matplotlib.use("Agg")  # écrit des images sans ouvrir de fenêtre

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle


# Flèche de chaque action (dx, dy) : 0=haut, 1=bas, 2=gauche, 3=droite.
# L'axe vertical est inversé (ligne 0 en haut), donc "haut" correspond à dy négatif.
FLECHES_LABYRINTHE = [(0, -0.3), (0, 0.3), (-0.3, 0), (0.3, 0)]
COULEURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # une couleur par série
COULEURS_ACTIONS_BRAS = ["#3b82f6", "#f97316", "#22c55e", "#a855f7"]
# Bras à 9 actions : la meilleure action se décompose en (variation de theta1,
# variation de theta2), chacune dans {-pas, 0, +pas}. Échelle divergente :
# rouge = diminue, gris clair = immobile, bleu = augmente.
COULEURS_VARIATION = ["#e34948", "#f0efec", "#2a78d6"]
# Valeurs d'epsilon (ordonnées) : dégradé bleu du clair au foncé.
BLEUS_ORDONNES = ["#86b6ef", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
ENCRE = "#0b0b0b"
ENCRE_SECONDAIRE = "#52514e"
GRILLE = "#e1e0d9"
FOND = "#fcfcfb"
MUR = "#52514e"
PIEGE = "#d03b3b"
ARRIVEE = "#0ca30c"
SANS_DONNEE = "#c3c2b7"  # gris pour une action impossible ou jamais essayée
ECHEC = "#f3b89a"  # orange pâle : la politique gloutonne tourne en rond

# Magnitude : une seule teinte, du clair au foncé. Gris : pas de donnée.
BLEUS = LinearSegmentedColormap.from_list(
    "bleus", ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
).with_extremes(bad=GRILLE)
# Même dégradé bleu pour les Q-values, mais gris spécifique aux données absentes.
BLEUS_Q = BLEUS.with_extremes(bad=SANS_DONNEE)

plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "figure.facecolor": FOND,
    "axes.facecolor": FOND,
    "savefig.facecolor": FOND,
    "text.color": ENCRE,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": ENCRE_SECONDAIRE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": matplotlib.cycler(color=COULEURS),
    "xtick.color": ENCRE_SECONDAIRE,
    "ytick.color": ENCRE_SECONDAIRE,
    "lines.linewidth": 2,
    "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# Petites fonctions utilitaires
# ---------------------------------------------------------------------------
def chemin_glouton(algorithme, environnement, etat=None):
    """Suit toujours la meilleure action depuis `etat`, sans exploration.

    Par défaut, part du départ de référence. Renvoie (chemin, retour, objectif_atteint).
    S'arrête si l'agent tourne en rond.
    """
    # Pas de reset() : sur le bras, il tirerait un départ et changerait l'entraînement.
    etat = depart(environnement) if etat is None else etat
    environnement.etat = etat
    chemin = [etat]
    retour = 0.0

    for _ in range(algorithme.max_pas):
        if not environnement.actions_possibles(etat):
            break

        action = algorithme.meilleure_action(etat)
        nouvel_etat, recompense, termine = environnement.step(action)
        retour += recompense
        deja_visite = nouvel_etat in chemin
        chemin.append(nouvel_etat)

        if termine:
            return chemin, retour, True
        if deja_visite:
            break

        etat = nouvel_etat

    return chemin, retour, False


def est_bras(environnement):
    return hasattr(environnement, "angles_theta1")


def disposition_actions(nb_actions):
    """(lignes, colonnes) de sous-graphes : 2 x 2 pour 4 actions, 3 x 3 pour 9."""
    cote = int(np.ceil(np.sqrt(nb_actions)))
    return int(np.ceil(nb_actions / cote)), cote


def dimensions(environnement):
    """Renvoie (lignes, colonnes) de la grille des états."""
    if est_bras(environnement):
        return len(environnement.angles_theta1), len(environnement.angles_theta2)
    return environnement.taille, environnement.taille


def depart(environnement):
    return environnement.etat_initial if est_bras(environnement) else environnement.depart


def arrivees(environnement):
    if est_bras(environnement):
        return list(environnement.etats_objectif)
    return [environnement.arrivee]


def cases_dangereuses(environnement):
    """Liste des pièges (5x5) ou des feux (9x9). Vide pour le 3x3 et le bras."""
    return getattr(environnement, "pieges", []) + getattr(environnement, "feux", [])


def petite_grille(environnement):
    """Vrai si chaque case est assez grande pour y écrire une valeur (5 x 5)."""
    return not est_bras(environnement) and environnement.taille <= 10


def moyenne_glissante(valeurs, fenetre):
    """Moyenne sur les `fenetre` derniers épisodes (fenêtre réduite au début)."""
    valeurs = np.asarray(valeurs, dtype=float)
    cumul = np.cumsum(np.insert(valeurs, 0, 0.0))
    fin = np.arange(1, len(valeurs) + 1)
    debut = np.maximum(0, fin - fenetre)
    return (cumul[fin] - cumul[debut]) / (fin - debut)


def en_grille(valeurs_par_etat, environnement):
    """Range un tableau indexé par état dans une grille (lignes, colonnes)."""
    return np.asarray(valeurs_par_etat, dtype=float).reshape(dimensions(environnement))


def grille_q(algorithme, environnement, action):
    """Grille des Q(s, a). NaN si Q(s, a) n'a jamais été mise à jour (mur, bord, arrivée)."""
    valeurs = np.where(algorithme.mises_a_jour[:, action] > 0, algorithme.Q[:, action], np.nan)
    return en_grille(valeurs, environnement)


def grille_valeurs(algorithme, environnement):
    """Grille de V(s) = max Q(s, a) sur les actions essayées. NaN si aucune (mur, arrivée...).

    Une action jamais essayée conserve sa Q-value initiale aléatoire :
    elle pourrait masquer dans le max des valeurs réellement apprises.
    """
    valeurs = np.where(algorithme.mises_a_jour > 0, algorithme.Q, -np.inf).max(axis=1)
    return en_grille(np.where(np.isfinite(valeurs), valeurs, np.nan), environnement)


def preparer_grille(ax, environnement):
    """Règle les axes pour afficher la grille des états, ligne 0 en haut."""
    lignes, colonnes = dimensions(environnement)
    ax.set_xlim(-0.5, colonnes - 0.5)
    ax.set_ylim(lignes - 0.5, -0.5)
    ax.set_aspect("equal")

    if est_bras(environnement):
        ax.set_xticks(range(0, colonnes, 6), environnement.angles_theta2[::6])
        ax.set_yticks(range(0, lignes, 6), environnement.angles_theta1[::6])
        ax.set_xlabel("theta2 (degrés)")
        ax.set_ylabel("theta1 (degrés)")
    elif not petite_grille(environnement):
        ax.set_xticks(range(0, colonnes, 5))
        ax.set_yticks(range(0, lignes, 5))
        ax.set_xlabel("colonne")
        ax.set_ylabel("ligne")
    else:
        ax.set_xticks(range(colonnes))
        ax.set_yticks(range(lignes))
        ax.set_xlabel("colonne")
        ax.set_ylabel("ligne")

    ax.set_xticks(np.arange(-0.5, colonnes), minor=True)
    ax.set_yticks(np.arange(-0.5, lignes), minor=True)
    ax.grid(which="minor", color=GRILLE, linewidth=1 if petite_grille(environnement) else 0.4)
    ax.tick_params(which="minor", length=0)


def colorier_case(ax, etat, colonnes, couleur):
    """Remplit la case `etat` avec une couleur."""
    ligne, colonne = divmod(etat, colonnes)
    ax.add_patch(Rectangle((colonne - 0.5, ligne - 0.5), 1, 1, color=couleur))


def entourer_case(ax, etat, colonnes, couleur):
    """Entoure la case `etat` sans masquer ce qu'elle contient."""
    ligne, colonne = divmod(etat, colonnes)
    ax.add_patch(Rectangle((colonne - 0.45, ligne - 0.45), 0.9, 0.9, fill=False,
                           edgecolor=couleur, linewidth=2, zorder=4))


def marquer_cases(ax, environnement):
    """Marque les arrivées et les obstacles ; S uniquement dans le labyrinthe."""
    colonnes = dimensions(environnement)[1]
    for etat in getattr(environnement, "murs", []):
        colorier_case(ax, etat, colonnes, MUR)
    for etat in cases_dangereuses(environnement):
        entourer_case(ax, etat, colonnes, PIEGE)
    for etat in arrivees(environnement):
        entourer_case(ax, etat, colonnes, ARRIVEE)

    # Le bras commence chaque épisode à une pose différente : pas de S unique.
    if not est_bras(environnement):
        ligne, colonne = divmod(depart(environnement), colonnes)
        ax.text(colonne - 0.32, ligne - 0.32, "S", ha="center", va="center",
                fontsize=8, fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", facecolor=FOND, edgecolor="none"))


def legende_cases(fig, environnement, gris, couleur_gris=GRILLE):
    """Légende sous la figure. `gris` : ce que signifie une case grise."""
    elements = [Patch(color=couleur_gris, label=gris)]
    if not est_bras(environnement):
        elements.insert(0, Line2D([], [], linestyle="none", marker="$S$",
                                  color=ENCRE, label="départ"))
    if getattr(environnement, "murs", []):
        elements.append(Patch(color=MUR, label="mur"))
    if cases_dangereuses(environnement):
        elements.append(Patch(fill=False, edgecolor=PIEGE, linewidth=2, label="piège"))
    elements.append(Patch(fill=False, edgecolor=ARRIVEE, linewidth=2, label="arrivée"))
    fig.legend(handles=elements, loc="outside lower center", ncols=len(elements))


def taille_carte(environnement):
    """Taille d'une figure contenant une seule carte."""
    if est_bras(environnement):
        return (10, 9)
    n = min(environnement.taille, 14)  # au-delà, les cases rétrécissent
    return (0.8 * n + 3, 0.8 * n + 1.5)


def afficher_comptages(ax, comptages, environnement, maxi, cmap=BLEUS, textes=None,
                       taille_texte=8):
    """Carte de comptages en échelle logarithmique ; zéro en gris.

    `textes` : grille de chaînes à écrire dans les cases (toutes tailles de grille).
    Par défaut, la valeur elle-même, seulement pour les petites grilles.
    """
    image = ax.imshow(np.where(comptages > 0, comptages, np.nan), cmap=cmap,
                      norm=LogNorm(vmin=1, vmax=max(2, maxi)))
    preparer_grille(ax, environnement)
    marquer_cases(ax, environnement)

    if textes is None and petite_grille(environnement):
        textes = [[f"{v:.0f}" if v > 0 else "" for v in ligne] for ligne in comptages]
    if textes is not None:
        for (ligne, colonne), valeur in np.ndenumerate(comptages):
            if textes[ligne][colonne]:
                ecrire_valeur(ax, ligne, colonne, textes[ligne][colonne],
                              image.norm(max(valeur, 1)), taille_texte)
    return image


def compact(nombre):
    """12 -> '12', 4321 -> '4.3k', 98765 -> '99k' : tient dans une petite case."""
    nombre = int(nombre)
    if nombre < 1000:
        return str(nombre)
    if nombre < 10_000:
        return f"{nombre / 1000:.1f}k"
    if nombre < 1_000_000:
        return f"{nombre / 1000:.0f}k"
    return f"{nombre / 1_000_000:.1f}M"


def ecrire_valeur(ax, ligne, colonne, texte, intensite, taille=8):
    """Écrit dans une case, en blanc sur les couleurs foncées."""
    ax.text(colonne, ligne, texte, ha="center", va="center", fontsize=taille, zorder=3,
            color="white" if intensite > 0.5 else ENCRE, linespacing=1.0)


# ---------------------------------------------------------------------------
# Graphe 3 : Q(s, a) brute en 2D, une carte par action
# ---------------------------------------------------------------------------
def graphe_cartes_q_brutes(algorithme, environnement, nom, fichier):
    grilles = [grille_q(algorithme, environnement, action)
               for action in range(environnement.nb_actions)]

    # Une même échelle linéaire pour les quatre cartes de cet environnement,
    # déterminée par les Q-values effectivement apprises, y compris négatives.
    observees = np.concatenate([grille[np.isfinite(grille)] for grille in grilles])
    if observees.size:
        minimum, maximum = float(observees.min()), float(observees.max())
        if np.isclose(minimum, maximum):
            minimum -= 0.5
            maximum += 0.5
    else:
        minimum, maximum = 0.0, 1.0

    lignes, colonnes = disposition_actions(environnement.nb_actions)
    taille = (11, 10) if petite_grille(environnement) else (6.5 * colonnes, 6 * lignes)
    fig, axes = plt.subplots(lignes, colonnes, figsize=taille, layout="constrained",
                             squeeze=False)

    for action, ax in enumerate(axes.flat[:environnement.nb_actions]):
        image = ax.imshow(grilles[action], cmap=BLEUS_Q, vmin=minimum, vmax=maximum)
        preparer_grille(ax, environnement)
        marquer_cases(ax, environnement)
        ax.set_title(f"action {action} : {environnement.noms_actions[action]}")

        if petite_grille(environnement):
            for (ligne, colonne), q in np.ndenumerate(grilles[action]):
                if np.isfinite(q):
                    ecrire_valeur(ax, ligne, colonne, f"{q:.2f}", image.norm(q))

    fig.colorbar(image, ax=axes, shrink=0.8,
                 label="Q(s, a) : bleu clair = valeur faible, bleu foncé = valeur élevée")
    fig.suptitle(f"{nom} : Q-values brutes par action")
    legende_cases(fig, environnement, "action impossible ou jamais essayée", SANS_DONNEE)
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graphe 4 : meilleure action dans chaque case et chemin glouton
# ---------------------------------------------------------------------------
def graphe_meilleure_action(algorithme, environnement, nom, fichier):
    colonnes = dimensions(environnement)[1]

    if est_bras(environnement) and environnement.nb_actions != 4:
        graphe_meilleure_action_9(algorithme, environnement, nom, fichier)
        return

    if est_bras(environnement):
        # Carte catégorielle : chaque couleur représente une action, pas une Q-value.
        # Aucun départ unique et aucun chemin de référence pour les bras.
        carte = np.full(dimensions(environnement), np.nan)
        for etat in range(environnement.nb_etats):
            if algorithme.visites[etat] == 0 or not environnement.actions_possibles(etat):
                continue
            ligne, colonne = divmod(etat, colonnes)
            carte[ligne, colonne] = algorithme.meilleure_action(etat)

        couleurs = ListedColormap(COULEURS_ACTIONS_BRAS).with_extremes(bad=SANS_DONNEE)
        normalisation = BoundaryNorm(np.arange(-0.5, 4.5, 1), couleurs.N)

        fig, ax = plt.subplots(figsize=(11, 10), layout="constrained")
        ax.imshow(carte, cmap=couleurs, norm=normalisation, interpolation="nearest")
        preparer_grille(ax, environnement)

        # Les configurations d'arrivée restent visibles sur toutes les couleurs.
        for etat in arrivees(environnement):
            ligne, colonne = divmod(etat, colonnes)
            ax.add_patch(Circle((colonne, ligne), radius=0.35, fill=False,
                                edgecolor=ENCRE, linewidth=4, zorder=5))
            ax.add_patch(Circle((colonne, ligne), radius=0.35, fill=False,
                                edgecolor="white", linewidth=2.2, zorder=6))

        legendes = [
            Patch(facecolor=couleur, label=nom_action.replace(" ", ""))
            for couleur, nom_action in zip(COULEURS_ACTIONS_BRAS, environnement.noms_actions)
        ]
        legendes.append(Patch(facecolor=SANS_DONNEE,
                              label="jamais visité / action impossible"))
        legendes.append(Line2D([], [], linestyle="none", marker="o", markersize=9,
                               markerfacecolor="none", markeredgecolor=ENCRE,
                               markeredgewidth=2, label="arrivée"))
        fig.legend(handles=legendes, loc="outside lower center", ncols=3)
        ax.set_title(f"{nom} : meilleure action par configuration angulaire")
        fig.savefig(fichier, dpi=150)
        plt.close(fig)
        return

    # Dans le labyrinthe, on conserve les flèches et le chemin glouton.
    chemin, retour, atteint = chemin_glouton(algorithme, environnement)
    fig, ax = plt.subplots(figsize=taille_carte(environnement), layout="constrained")
    preparer_grille(ax, environnement)

    for etat in range(environnement.nb_etats):
        if algorithme.visites[etat] == 0:
            colorier_case(ax, etat, colonnes, GRILLE)
        elif environnement.actions_possibles(etat):
            ligne, colonne = divmod(etat, colonnes)
            action = algorithme.meilleure_action(etat)
            dx, dy = FLECHES_LABYRINTHE[action]
            ax.arrow(colonne - dx, ligne - dy, 2 * dx, 2 * dy,
                     head_width=0.15 if petite_grille(environnement) else 0.25,
                     length_includes_head=True,
                     color=ENCRE, linewidth=0.5 if petite_grille(environnement) else 0.3)

    marquer_cases(ax, environnement)

    x, y = [], []
    for precedent, etat in zip(chemin[:1] + chemin, chemin):
        ligne_avant, colonne_avant = divmod(precedent, colonnes)
        ligne, colonne = divmod(etat, colonnes)
        if abs(ligne - ligne_avant) + abs(colonne - colonne_avant) > 1:
            x.append(np.nan)
            y.append(np.nan)
        x.append(colonne)
        y.append(ligne)

    ax.plot(x, y, color=COULEURS[1], linewidth=8 if petite_grille(environnement) else 5,
            alpha=0.35 if petite_grille(environnement) else 0.6,
            solid_capstyle="round", zorder=2)

    statut = "objectif atteint" if atteint else "objectif non atteint"
    ax.set_title(f"{nom} : meilleure action argmax Q(s, a) et chemin glouton\n"
                 f"{len(chemin) - 1} déplacements, retour {retour:.2f}, {statut}")
    legende_cases(fig, environnement, "état jamais visité")
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


def entourer_arrivees_bras(ax, environnement):
    """Cercles blancs bordés de noir, visibles sur toutes les couleurs."""
    colonnes = dimensions(environnement)[1]
    for etat in arrivees(environnement):
        ligne, colonne = divmod(etat, colonnes)
        ax.add_patch(Circle((colonne, ligne), radius=0.35, fill=False,
                            edgecolor=ENCRE, linewidth=4, zorder=5))
        ax.add_patch(Circle((colonne, ligne), radius=0.35, fill=False,
                            edgecolor="white", linewidth=2.2, zorder=6))


def graphe_meilleure_action_9(algorithme, environnement, nom, fichier):
    """Bras à 9 actions : deux cartes, une par articulation.

    Neuf couleurs catégorielles ne se distinguent pas ; on décompose donc la
    meilleure action en (variation de theta1, variation de theta2), chacune
    codée sur trois niveaux (diminue, immobile, augmente).
    """
    colonnes = dimensions(environnement)[1]
    variations = np.full((2,) + dimensions(environnement), np.nan)
    nb_simultanees = nb_immobiles = nb_etats = 0
    for etat in range(environnement.nb_etats):
        if algorithme.visites[etat] == 0 or not environnement.actions_possibles(etat):
            continue
        d1, d2 = environnement.deplacements[algorithme.meilleure_action(etat)]
        ligne, colonne = divmod(etat, colonnes)
        variations[:, ligne, colonne] = (d1, d2)
        nb_etats += 1
        nb_simultanees += d1 != 0 and d2 != 0
        nb_immobiles += d1 == 0 and d2 == 0

    couleurs = ListedColormap(COULEURS_VARIATION).with_extremes(bad=SANS_DONNEE)
    normalisation = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], couleurs.N)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10.5), layout="constrained")
    for indice, ax in enumerate(axes):
        ax.imshow(variations[indice], cmap=couleurs, norm=normalisation,
                  interpolation="nearest")
        preparer_grille(ax, environnement)
        entourer_arrivees_bras(ax, environnement)
        ax.set_title(f"variation de theta{indice + 1} choisie")

    legendes = [Patch(facecolor=COULEURS_VARIATION[2], label="+pas (augmente)"),
                Patch(facecolor=COULEURS_VARIATION[1], edgecolor=GRILLE,
                      label="0 (articulation immobile)"),
                Patch(facecolor=COULEURS_VARIATION[0], label="-pas (diminue)"),
                Patch(facecolor=SANS_DONNEE, label="jamais visité / action impossible"),
                Line2D([], [], linestyle="none", marker="o", markersize=9,
                       markerfacecolor="none", markeredgecolor=ENCRE,
                       markeredgewidth=2, label="arrivée")]
    fig.legend(handles=legendes, loc="outside lower center", ncols=len(legendes))
    part = (lambda n: 100 * n / nb_etats) if nb_etats else (lambda n: 0.0)
    fig.suptitle(f"{nom} : meilleure action par configuration angulaire\n"
                 f"les deux articulations bougent dans {part(nb_simultanees):.0f} % "
                 f"des états visités, aucune dans {part(nb_immobiles):.0f} %")
    fig.savefig(fichier, dpi=150)
    plt.close(fig)
    return {"etats": nb_etats, "simultanees": nb_simultanees, "immobiles": nb_immobiles}


# ---------------------------------------------------------------------------
# Graphe 6 : nombre de passages par chaque état
# ---------------------------------------------------------------------------
def graphe_visites(algorithme, environnement, nom, fichier):
    """Couleur : passages. Texte de chaque case : passages / mises à jour de Q(s, .).

    Les deux nombres diffèrent : un passage compte chaque arrivée dans l'état
    (et le départ de chaque épisode), une mise à jour chaque action jouée depuis
    l'état. L'arrivée est visitée mais jamais mise à jour.
    """
    visites = en_grille(algorithme.visites, environnement)
    mises_a_jour = en_grille(algorithme.mises_a_jour.sum(axis=1), environnement)
    formater = (lambda n: f"{int(n)}") if petite_grille(environnement) else compact
    textes = [[f"{formater(v)}\n{formater(m)}" if v > 0 else ""
               for v, m in zip(ligne_v, ligne_m)]
              for ligne_v, ligne_m in zip(visites, mises_a_jour)]

    if petite_grille(environnement):
        taille, taille_texte = taille_carte(environnement), 8
    else:  # 36 x 36 : grande image pour que chaque case reste lisible au zoom
        taille, taille_texte = (30, 28), 5
    fig, ax = plt.subplots(figsize=taille, layout="constrained")
    image = afficher_comptages(ax, visites, environnement, visites.max(),
                               textes=textes, taille_texte=taille_texte)

    fig.colorbar(image, ax=ax, shrink=0.8, format="%d", label="passages (échelle log)")
    ax.set_title(f"{nom} : nombre de passages par état\n"
                 f"dans chaque case : passages (haut) / mises à jour de Q (bas)\n"
                 f"{int(visites.sum())} passages, {int(mises_a_jour.sum())} mises à jour, "
                 f"{np.count_nonzero(visites)} états visités sur {visites.size}")
    legende_cases(fig, environnement, "jamais visité")
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graphe 6b : nombre de passages par type de déplacement (une couleur par action)
# ---------------------------------------------------------------------------
def graphe_passages_par_action(algorithme, environnement, nom, fichier):
    """Combien de fois chaque action a été jouée depuis chaque état.

    Labyrinthe : haut, bas, gauche, droite. Bras : theta1+, theta1-, theta2+,
    theta2-, sur la grille des couples (theta1, theta2). Échelle log commune
    aux quatre cartes ; gris = action jamais jouée depuis cet état.
    """
    comptages = [en_grille(algorithme.mises_a_jour[:, action], environnement)
                 for action in range(environnement.nb_actions)]
    maxi = max(float(c.max()) for c in comptages)
    total = algorithme.mises_a_jour.sum()

    lignes, colonnes = disposition_actions(environnement.nb_actions)
    taille = (12, 11) if petite_grille(environnement) else (7.5 * colonnes, 7 * lignes)
    fig, axes = plt.subplots(lignes, colonnes, figsize=taille, layout="constrained",
                             squeeze=False)
    for action, ax in enumerate(axes.flat[:environnement.nb_actions]):
        # Même palette bleue et même échelle pour toutes les cartes : comparables.
        image = afficher_comptages(ax, comptages[action], environnement, maxi)
        part = 100 * comptages[action].sum() / total if total else 0
        ax.set_title(f"{environnement.noms_actions[action]} : "
                     f"{int(comptages[action].sum())} fois ({part:.0f} %)")
    fig.colorbar(image, ax=axes, shrink=0.8, format="%d",
                 label="nombre de fois où l'action a été jouée depuis la case (log)")

    fig.suptitle(f"{nom} : nombre de passages par type de déplacement")
    legende_cases(fig, environnement, "action jamais jouée depuis cet état")
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Indicateurs de synthèse affichés dans le terminal
# ---------------------------------------------------------------------------
def indicateurs(algorithme, environnement):
    """Mesures comparables entre environnements, dont les récompenses diffèrent."""
    etats = environnement.nb_etats - len(getattr(environnement, "murs", []))
    couples = sum(len(environnement.actions_possibles(e)) for e in range(environnement.nb_etats))
    visites = np.sort(algorithme.visites)[::-1]
    mises_a_jour = algorithme.mises_a_jour
    chemin, _, atteint = chemin_glouton(algorithme, environnement)
    departs = [e for e in range(environnement.nb_etats) if environnement.actions_possibles(e)]
    reussites = sum(chemin_glouton(algorithme, environnement, e)[2] for e in departs)

    return {
        "etats": etats,
        "couples": couples,
        "mises_a_jour": mises_a_jour.sum(),
        "mises_a_jour_par_couple": mises_a_jour.sum() / couples,
        "etats_visites": 100 * np.count_nonzero(visites) / etats,
        "couples_jamais_essayes": 100 * (1 - np.count_nonzero(mises_a_jour) / couples),
        "visites_top_10": 100 * visites[:max(1, etats // 10)].sum() / visites.sum(),
        "exploration": 100 * algorithme.explorations.sum() / mises_a_jour.sum(),
        "reussite_tous_departs": 100 * reussites / len(departs),
        "deplacements": len(chemin) - 1,
        "atteint": atteint,
    }


# ---------------------------------------------------------------------------
# Comparaison des récompenses du bras : progression de la réussite
# ---------------------------------------------------------------------------
def graphe_progression_reussite(suivis, fichier,
                                titre="Bras robotique : vitesse d'apprentissage "
                                      "des trois récompenses"):
    """Compare le succès glouton à chaque point d'évaluation (mêmes départs)."""
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    for i, (mode, bilans) in enumerate(suivis.items()):
        episodes = [bilan["episode"] for bilan in bilans]
        taux = [bilan["taux_reussite"] for bilan in bilans]
        ax.plot(episodes, taux, marker="o", markersize=5, linewidth=2.2,
                label=mode.capitalize(), color=COULEURS[i % len(COULEURS)])

    bilans_reference = next(iter(suivis.values()))
    ax.set_xticks([bilan["episode"] for bilan in bilans_reference])
    ax.set_xlim(0, bilans_reference[-1]["episode"] * 1.02)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Nombre d'épisodes d'entraînement")
    ax.set_ylabel("Taux de réussite (%)")
    ax.grid(True, alpha=0.2)
    ax.legend(title="Récompense", loc="upper left")
    ax.set_title(titre)
    fig.suptitle(
        f"Politique gloutonne, mêmes {bilans_reference[-1]['tests']} départs de test, "
        f"{bilans_reference[-1]['max_pas']} mouvements maximum par essai",
        fontsize=10, color=ENCRE_SECONDAIRE,
    )
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Politique optimale du labyrinthe (référence exacte, calculée sans apprentissage)
# ---------------------------------------------------------------------------
def retours_optimaux(environnement):
    """Meilleur retour non actualisé possible depuis chaque état (Dijkstra).

    Le coût d'entrée dans une case vaut -recompense(case) ; il est positif
    partout sauf à l'arrivée, qui termine l'épisode. NaN pour les murs,
    l'arrivée elle-même et les cases d'où l'arrivée est inatteignable.
    """
    n = environnement.nb_etats
    predecesseurs = [[] for _ in range(n)]
    for etat in range(n):
        for action in environnement.actions_possibles(etat):
            predecesseurs[environnement.etat_suivant(etat, action)].append(etat)

    # cout[s] = somme des -r(v) sur les cases intermédiaires entre s et l'arrivée.
    arrivee = environnement.arrivee
    cout = np.full(n, np.inf)
    file = []
    for etat in predecesseurs[arrivee]:
        cout[etat] = 0.0
        file.append((0.0, etat))
    heapq.heapify(file)
    while file:
        c, v = heapq.heappop(file)
        if c > cout[v]:
            continue
        entree = -environnement.recompense(v)
        for u in predecesseurs[v]:
            if u != arrivee and c + entree < cout[u]:
                cout[u] = c + entree
                heapq.heappush(file, (cout[u], u))

    retour = environnement.recompense(arrivee) - cout
    retour[np.isinf(cout)] = np.nan
    return retour


# ---------------------------------------------------------------------------
# Graphe 7 : écart entre la politique apprise et la politique optimale
# ---------------------------------------------------------------------------
def graphe_ecart_optimal(algorithme, environnement, nom, fichier):
    """Pour chaque départ : retour optimal - retour du chemin glouton appris.

    0 (bleu très clair) = la politique apprise est optimale depuis cette case.
    Orange : la politique gloutonne n'atteint pas l'arrivée (boucle ou limite).
    """
    optimal = retours_optimaux(environnement)
    ecart = np.full(environnement.nb_etats, np.nan)
    echecs = []
    etat_avant = environnement.etat
    for etat in range(environnement.nb_etats):
        if not environnement.actions_possibles(etat) or np.isnan(optimal[etat]):
            continue
        _, retour, atteint = chemin_glouton(algorithme, environnement, etat)
        if atteint:
            ecart[etat] = max(0.0, optimal[etat] - retour)
        else:
            echecs.append(etat)
    environnement.etat = etat_avant

    colonnes = dimensions(environnement)[1]
    valides = ecart[np.isfinite(ecart)]
    nb_optimaux = int(np.sum(np.isclose(valides, 0.0)))
    total = len(valides) + len(echecs)

    fig, ax = plt.subplots(figsize=taille_carte(environnement), layout="constrained")
    maxi = max(0.5, float(valides.max()) if valides.size else 0.5)
    grille = en_grille(ecart, environnement)
    image = ax.imshow(grille, cmap=BLEUS, vmin=0, vmax=maxi)
    preparer_grille(ax, environnement)
    for etat in echecs:
        colorier_case(ax, etat, colonnes, ECHEC)
    marquer_cases(ax, environnement)

    # Trajet réellement appris depuis S : les cases réussies l'entourent.
    chemin, _, _ = chemin_glouton(algorithme, environnement)
    environnement.etat = etat_avant
    lignes_chemin, colonnes_chemin = zip(*(divmod(e, colonnes) for e in chemin))
    ax.plot(colonnes_chemin, lignes_chemin, color=ENCRE, linewidth=2, zorder=4)

    if petite_grille(environnement):
        for (ligne, colonne), valeur in np.ndenumerate(grille):
            if np.isfinite(valeur):
                ecrire_valeur(ax, ligne, colonne, f"{valeur:.1f}", image.norm(valeur))

    fig.colorbar(image, ax=ax, shrink=0.8,
                 label="retour optimal - retour obtenu (0 = chemin optimal)")
    ax.set_title(f"{nom} : qualité de la politique apprise si l'agent part de chaque case\n"
                 f"{nb_optimaux}/{total} cases : chemin optimal ; "
                 f"{total - nb_optimaux - len(echecs)} : arrivée atteinte mais détour ; "
                 f"{len(echecs)} : boucle, arrivée jamais atteinte")
    elements = [Patch(color=ECHEC, label="politique en boucle (case peu apprise)"),
                Line2D([], [], color=ENCRE, linewidth=2, label="chemin appris depuis S")]
    if getattr(environnement, "murs", []):
        elements.append(Patch(color=MUR, label="mur"))
    if cases_dangereuses(environnement):
        elements.append(Patch(fill=False, edgecolor=PIEGE, linewidth=2, label="piège"))
    elements.append(Patch(fill=False, edgecolor=ARRIVEE, linewidth=2, label="arrivée"))
    fig.legend(handles=elements, loc="outside lower center", ncols=len(elements))
    fig.savefig(fichier, dpi=150)
    plt.close(fig)
    return {"optimaux": nb_optimaux, "echecs": len(echecs), "departs": total}


# ---------------------------------------------------------------------------
# Comparaison des labyrinthes 5 x 5 et 36 x 36
# ---------------------------------------------------------------------------
def graphe_comparaison_labyrinthes(suivis, fichier):
    """Compare l'apprentissage de labyrinthes de tailles différentes.

    `suivis` : {nom: {"algorithme", "evaluations", "optimal"}}. L'axe horizontal
    est la progression de l'entraînement (%), car les labyrinthes n'ont pas
    le même nombre d'épisodes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), layout="constrained")
    (ax_long, ax_reussite), (ax_couverture, ax_depart) = axes

    for i, (nom, suivi) in enumerate(suivis.items()):
        algorithme = suivi["algorithme"]
        couleur = COULEURS[i % len(COULEURS)]
        n = len(algorithme.nombre_actions)
        progression = 100 * np.arange(1, n + 1) / n
        fenetre = max(10, n // 50)
        etiquette = f"{nom} ({n} épisodes, max {algorithme.max_pas} pas)"

        rapport = moyenne_glissante(algorithme.nombre_actions, fenetre) / suivi["optimal"]
        ax_long.plot(progression, rapport, color=couleur, label=etiquette)

        evaluations = suivi["evaluations"]
        x = [100 * e["episode"] / n for e in evaluations]
        ax_reussite.plot(x, [e["taux_reussite"] for e in evaluations], color=couleur,
                         marker="o", markersize=5, label=etiquette)
        ax_couverture.plot(x, [e["taux_optimal"] for e in evaluations], color=couleur,
                           marker="o", markersize=5, label=etiquette)
        # Chemin glouton depuis le départ ; trou dans la courbe tant qu'il échoue.
        rapports = [np.nan if e["longueur_depart"] is None
                    else e["longueur_depart"] / suivi["optimal"] for e in evaluations]
        ax_depart.plot(x, rapports, color=couleur, marker="o", markersize=5, label=etiquette)

    ax_long.axhline(1, color=ENCRE_SECONDAIRE, linewidth=1, linestyle="--")
    ax_long.text(1, 1, "plus court chemin", ha="left", va="bottom",
                 fontsize=9, color=ENCRE_SECONDAIRE)
    ax_long.set_yscale("log")
    ax_long.set_ylabel("longueur d'épisode / plus court chemin\n(moyenne glissante, log)")
    ax_long.set_title("Longueur des épisodes, normalisée par le plus court chemin")

    ax_reussite.set_ylim(0, 105)
    ax_reussite.set_ylabel("cases de départ menant à l'arrivée (%)")
    ax_reussite.set_title("Réussite de la politique gloutonne depuis toutes les cases")

    ax_couverture.set_ylim(0, 105)
    ax_couverture.set_ylabel("cases de départ avec chemin optimal (%)")
    ax_couverture.set_title("Politique optimale (meilleur retour possible, pièges compris)")

    ax_depart.axhline(1, color=ENCRE_SECONDAIRE, linewidth=1, linestyle="--")
    ax_depart.set_ylim(0.9, None)
    ax_depart.set_ylabel("chemin glouton / plus court chemin")
    ax_depart.set_title("Chemin appris depuis le départ S (absent = arrivée non atteinte)")

    for ax in axes.flat:
        ax.set_xlim(0, 101)
        ax.set_xlabel("progression de l'entraînement (%)")
        ax.grid(True, alpha=0.2)
    fig.legend(*ax_reussite.get_legend_handles_labels(),
               loc="outside lower center", ncols=len(suivis))
    fig.suptitle("Labyrinthes 5 x 5 et 36 x 36 : effet de la taille sur le Q-learning")
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bras : longueur des épisodes pendant l'entraînement (comparable entre récompenses)
# ---------------------------------------------------------------------------
def graphe_longueur_episodes_bras(algorithmes, fichier, fenetre=100,
                                  titre="Bras robotique : longueur des épisodes "
                                        "pendant l'apprentissage"):
    """`algorithmes` : {mode: QLearning}. Les échelles de récompense diffèrent,
    mais le nombre de mouvements par épisode est comparable."""
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    max_pas, nb_episodes = 0, 0
    for i, (mode, algorithme) in enumerate(algorithmes.items()):
        couleur = COULEURS[i % len(COULEURS)]
        episodes = np.arange(1, len(algorithme.nombre_actions) + 1)
        moyenne = moyenne_glissante(algorithme.nombre_actions, fenetre)
        ax.plot(episodes, moyenne, color=couleur, label=mode.capitalize())
        ax.text(episodes[-1] * 1.01, moyenne[-1], mode.capitalize(),
                va="center", fontsize=9, color=ENCRE)
        max_pas = max(max_pas, algorithme.max_pas)
        nb_episodes = max(nb_episodes, len(episodes))

    ax.axhline(max_pas, color=ENCRE_SECONDAIRE, linewidth=1, linestyle="--")
    ax.text(0, max_pas, f" limite : {max_pas} mouvements (épisode sans succès)",
            va="bottom", fontsize=9, color=ENCRE_SECONDAIRE)
    ax.set_xlim(0, nb_episodes * 1.1)
    ax.set_ylim(0, max_pas * 1.12)
    ax.set_xlabel("épisode d'entraînement")
    ax.set_ylabel(f"mouvements par épisode (moyenne glissante sur {fenetre})")
    ax.grid(True, alpha=0.2)
    ax.legend(title="Récompense", loc="lower left")
    ax.set_title(f"{titre}\n(exploration epsilon-greedy incluse)")
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bras : 4 actions (une articulation à la fois) contre 9 actions (simultanées)
# ---------------------------------------------------------------------------
def graphe_comparaison_actions(suivis, algorithmes, fichier, fenetre=100):
    """`suivis[nb_actions][mode]` : bilans d'évaluation ; `algorithmes[nb_actions][mode]`.

    Couleur = récompense, trait plein = 4 actions, tirets = 9 actions.
    """
    styles = {4: "-", 9: "--"}
    fig, (ax_reussite, ax_longueur) = plt.subplots(1, 2, figsize=(16, 6.5),
                                                   layout="constrained")
    max_pas = 0
    for nb_actions, suivis_modes in suivis.items():
        for i, (mode, bilans) in enumerate(suivis_modes.items()):
            couleur = COULEURS[i % len(COULEURS)]
            etiquette = f"{mode.capitalize()}, {nb_actions} actions"
            ax_reussite.plot([b["episode"] for b in bilans],
                             [b["taux_reussite"] for b in bilans],
                             linestyle=styles.get(nb_actions, ":"), marker="o",
                             markersize=5, color=couleur, label=etiquette)
            algorithme = algorithmes[nb_actions][mode]
            episodes = np.arange(1, len(algorithme.nombre_actions) + 1)
            ax_longueur.plot(episodes, moyenne_glissante(algorithme.nombre_actions, fenetre),
                             linestyle=styles.get(nb_actions, ":"), color=couleur,
                             label=etiquette)
            max_pas = max(max_pas, algorithme.max_pas)

    ax_reussite.set_ylim(0, 105)
    ax_reussite.set_xlabel("Nombre d'épisodes d'entraînement")
    ax_reussite.set_ylabel("Taux de réussite (%)")
    ax_reussite.set_title("Réussite de la politique gloutonne (mêmes départs de test)")
    ax_longueur.axhline(max_pas, color=ENCRE_SECONDAIRE, linewidth=1, linestyle=":")
    ax_longueur.set_ylim(0, max_pas * 1.08)
    ax_longueur.set_xlabel("épisode d'entraînement")
    ax_longueur.set_ylabel(f"mouvements par épisode (moyenne glissante sur {fenetre})")
    ax_longueur.set_title("Longueur des épisodes (exploration incluse)")
    for ax in (ax_reussite, ax_longueur):
        ax.grid(True, alpha=0.2)
    fig.legend(*ax_reussite.get_legend_handles_labels(), loc="outside lower center",
               ncols=len(suivis))
    fig.suptitle("Bras robotique : une articulation à la fois (4 actions) "
                 "ou les deux en même temps (9 actions)")
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bras : impact d'epsilon sur l'exploration, initialisation constante / aléatoire
# ---------------------------------------------------------------------------
NOMS_INITIALISATIONS = {"constante": "Q initiale constante",
                        "aleatoire": "Q initiale aléatoire"}


def graphe_epsilon_evolution(resultats, cle, etiquette_y, titre, fichier):
    """Une ligne par récompense, une colonne par initialisation, une courbe par epsilon.

    `resultats[mode][initialisation][epsilon]["evaluations"]` : bilans successifs.
    """
    modes = list(resultats)
    initialisations = list(resultats[modes[0]])
    epsilons = list(resultats[modes[0]][initialisations[0]])
    fig, axes = plt.subplots(len(modes), len(initialisations),
                             figsize=(6.5 * len(initialisations), 3.6 * len(modes) + 1),
                             layout="constrained", sharex=True, sharey=True, squeeze=False)
    for ligne, mode in enumerate(modes):
        for colonne, initialisation in enumerate(initialisations):
            ax = axes[ligne, colonne]
            for i, epsilon in enumerate(epsilons):
                bilans = resultats[mode][initialisation][epsilon]["evaluations"]
                ax.plot([b["episode"] for b in bilans], [b[cle] for b in bilans],
                        color=BLEUS_ORDONNES[min(i, len(BLEUS_ORDONNES) - 1)],
                        label=f"epsilon = {epsilon:g}")
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.2)
            nom_init = NOMS_INITIALISATIONS.get(initialisation, initialisation)
            ax.set_title(f"{mode.capitalize()} - {nom_init}")
            if colonne == 0:
                ax.set_ylabel(etiquette_y)
            if ligne == len(modes) - 1:
                ax.set_xlabel("épisode d'entraînement")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center",
               ncols=len(epsilons))
    fig.suptitle(titre)
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


def graphe_epsilon_bilan(resultats, fichier):
    """Mesures finales en fonction d'epsilon : une colonne par récompense.

    Lignes : couples (s, a) essayés, réussite gloutonne, longueur des derniers épisodes.
    Bleu = Q initiale constante, orange = Q initiale aléatoire.
    """
    mesures = [("couples_essayes", "couples (s, a) essayés (%)", (0, 105)),
               ("taux_reussite", "réussite gloutonne (%)", (0, 105)),
               ("longueur_finale", "mouvements par épisode\n(500 derniers épisodes)", None)]
    modes = list(resultats)
    fig, axes = plt.subplots(len(mesures), len(modes),
                             figsize=(5.5 * len(modes), 3.4 * len(mesures) + 1),
                             layout="constrained", sharex=True, squeeze=False)
    for colonne, mode in enumerate(modes):
        for i, (initialisation, series) in enumerate(resultats[mode].items()):
            epsilons = list(series)
            positions = np.arange(len(epsilons))
            for ligne, (cle, etiquette, limites) in enumerate(mesures):
                ax = axes[ligne, colonne]
                ax.plot(positions, [series[e]["final"][cle] for e in epsilons],
                        marker="o", markersize=7, color=COULEURS[i % len(COULEURS)],
                        label=NOMS_INITIALISATIONS.get(initialisation, initialisation))
                ax.set_xticks(positions, [f"{e:g}" for e in epsilons])
                if limites:
                    ax.set_ylim(*limites)
                else:
                    ax.set_ylim(bottom=0)
                ax.grid(True, alpha=0.2)
                if colonne == 0:
                    ax.set_ylabel(etiquette)
        axes[0, colonne].set_title(mode.capitalize())
        axes[-1, colonne].set_xlabel("epsilon (constant pendant tout l'entraînement)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center",
               ncols=2)
    fig.suptitle("Bras robotique : impact d'epsilon sur l'exploration et la politique apprise")
    fig.savefig(fichier, dpi=150)
    plt.close(fig)
