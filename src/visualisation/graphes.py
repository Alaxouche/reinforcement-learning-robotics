"""Fonctions de tracé des graphes d'analyse du Q-learning (matplotlib).

Elles acceptent un labyrinthe comme le bras robotique : dans les deux cas, les
états forment une grille et etat = ligne * nb_colonnes + colonne. Pour le bras,
une ligne est un angle theta1 et une colonne un angle theta2.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # écrit des images sans ouvrir de fenêtre

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


# Flèche de chaque action (dx, dy) : 0=haut, 1=bas, 2=gauche, 3=droite.
# L'axe vertical est inversé (ligne 0 en haut), donc "haut" correspond à dy négatif.
FLECHES_LABYRINTHE = [(0, -0.3), (0, 0.3), (-0.3, 0), (0.3, 0)]
COULEURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # une couleur par série
ENCRE = "#0b0b0b"
ENCRE_SECONDAIRE = "#52514e"
GRILLE = "#e1e0d9"
FOND = "#fcfcfb"
MUR = "#52514e"
PIEGE = "#d03b3b"
ARRIVEE = "#0ca30c"
SANS_DONNEE = "#c3c2b7"  # gris plus soutenu, le zéro de ROUGE_BLEU étant déjà gris clair

# Magnitude : une seule teinte, du clair au foncé. Gris : pas de donnée.
BLEUS = LinearSegmentedColormap.from_list(
    "bleus", ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
).with_extremes(bad=GRILLE)
ORANGES = LinearSegmentedColormap.from_list(
    "oranges", ["#fbe3d8", "#f4a988", "#eb6834", "#b8481c", "#7a2c0e"]
).with_extremes(bad=GRILLE)
# Signe : rouge = valeur négative, gris clair = zéro, bleu = valeur positive.
ROUGE_BLEU = LinearSegmentedColormap.from_list(
    "rouge_bleu", ["#a32b2a", "#e34948", "#f0efec", "#2a78d6", "#104281"]
).with_extremes(bad=SANS_DONNEE)

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


def en_grille(valeurs_par_etat, environnement):
    """Range un tableau indexé par état dans une grille (lignes, colonnes)."""
    return np.asarray(valeurs_par_etat, dtype=float).reshape(dimensions(environnement))


def grille_q(algorithme, environnement, action):
    """Grille des Q(s, a). NaN si Q(s, a) n'a jamais été mise à jour (mur, bord, arrivée)."""
    valeurs = np.where(algorithme.mises_a_jour[:, action] > 0, algorithme.Q[:, action], np.nan)
    return en_grille(valeurs, environnement)


def grille_valeurs(algorithme, environnement):
    """Grille de V(s) = max Q(s, a) sur les actions essayées. NaN si aucune (mur, arrivée...).

    Une action jamais essayée garde son Q initial de 0 : avec des récompenses
    négatives, elle masquerait dans le max tout ce que l'état a appris.
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
    else:
        ax.set_xticks(range(colonnes))
        ax.set_yticks(range(lignes))
        ax.set_xlabel("colonne")
        ax.set_ylabel("ligne")

    ax.set_xticks(np.arange(-0.5, colonnes), minor=True)
    ax.set_yticks(np.arange(-0.5, lignes), minor=True)
    ax.grid(which="minor", color=GRILLE, linewidth=0.4 if est_bras(environnement) else 1)
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
    n = environnement.taille
    return (0.8 * n + 3, 0.8 * n + 1.5)


def afficher_comptages(ax, comptages, environnement, maxi):
    """Carte de comptages avec une échelle de couleur linéaire de 0 au maximum."""
    image = ax.imshow(np.where(comptages > 0, comptages, np.nan), cmap=BLEUS,
                      vmin=0, vmax=max(1, maxi))
    preparer_grille(ax, environnement)
    marquer_cases(ax, environnement)

    if not est_bras(environnement):
        for (ligne, colonne), valeur in np.ndenumerate(comptages):
            if valeur > 0:
                ecrire_valeur(ax, ligne, colonne, f"{valeur:.0f}", image.norm(valeur))
    return image


def ecrire_valeur(ax, ligne, colonne, texte, intensite):
    """Écrit dans une case, en blanc sur les couleurs foncées."""
    ax.text(colonne, ligne, texte, ha="center", va="center", fontsize=8, zorder=3,
            color="white" if intensite > 0.5 else ENCRE)


# ---------------------------------------------------------------------------
# Graphe 3 : Q(s, a) brute en 2D, une carte par action
# ---------------------------------------------------------------------------
def graphe_cartes_q_brutes(algorithme, environnement, nom, fichier):
    grilles = [grille_q(algorithme, environnement, action)
               for action in range(environnement.nb_actions)]
    borne = np.nanmax(np.abs(grilles))  # même échelle pour les 4 cartes, 0 au centre

    taille = (12, 11) if est_bras(environnement) else (11, 10)
    fig, axes = plt.subplots(2, 2, figsize=taille, layout="constrained")

    for action, ax in enumerate(axes.flat):
        image = ax.imshow(grilles[action], cmap=ROUGE_BLEU, vmin=-borne, vmax=borne)
        preparer_grille(ax, environnement)
        marquer_cases(ax, environnement)
        ax.set_title(f"action {action} : {environnement.noms_actions[action]}")

        if not est_bras(environnement):
            for (ligne, colonne), q in np.ndenumerate(grilles[action]):
                if np.isfinite(q):
                    ecrire_valeur(ax, ligne, colonne, f"{q:.2f}", abs(q) / borne)

    fig.colorbar(image, ax=axes, shrink=0.8, label="Q(s, a) : rouge = négatif, bleu = positif")
    fig.suptitle(f"{nom} : Q-values brutes par action")
    legende_cases(fig, environnement, "action impossible ou jamais essayée", SANS_DONNEE)
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graphe 4 : meilleure action dans chaque case et chemin glouton
# ---------------------------------------------------------------------------
def graphe_meilleure_action(algorithme, environnement, nom, fichier):
    colonnes = dimensions(environnement)[1]
    chemin, retour, atteint = chemin_glouton(algorithme, environnement)
    # Une grille 36 x 36 avec des noms d'actions demande davantage de place.
    taille = (14, 13) if est_bras(environnement) else taille_carte(environnement)
    fig, ax = plt.subplots(figsize=taille, layout="constrained")
    preparer_grille(ax, environnement)

    # Une flèche par état : la meilleure action. Un état jamais visité n'a rien appris.
    for etat in range(environnement.nb_etats):
        if algorithme.visites[etat] == 0:
            colorier_case(ax, etat, colonnes, GRILLE)
        elif environnement.actions_possibles(etat):
            ligne, colonne = divmod(etat, colonnes)
            action = algorithme.meilleure_action(etat)
            if est_bras(environnement):
                # Au lieu de 1 296 flèches, écrire la rotation optimale par case.
                etiquette = environnement.noms_actions[action].replace(" ", "")
                ax.text(colonne, ligne, etiquette, ha="center", va="center",
                        fontsize=4.5, fontweight="bold", color=ENCRE, zorder=4)
            else:
                dx, dy = FLECHES_LABYRINTHE[action]
                ax.arrow(colonne - dx, ligne - dy, 2 * dx, 2 * dy,
                         head_width=0.15, length_includes_head=True,
                         color=ENCRE, linewidth=0.5)

    marquer_cases(ax, environnement)

    # Chemin glouton : on relie les centres des cases visitées. Sur le bras, un angle
    # repasse de 350 à 0 degrés : on coupe le trait au lieu de traverser la grille.
    x, y = [], []
    for precedent, etat in zip(chemin[:1] + chemin, chemin):
        ligne_avant, colonne_avant = divmod(precedent, colonnes)
        ligne, colonne = divmod(etat, colonnes)
        if abs(ligne - ligne_avant) + abs(colonne - colonne_avant) > 1:
            x.append(np.nan)
            y.append(np.nan)
        x.append(colonne)
        y.append(ligne)
    # Le chemin doit rester visible sans masquer les noms d'actions.
    ax.plot(x, y, color=COULEURS[1], linewidth=8, alpha=0.35,
            solid_capstyle="round", zorder=2)

    statut = "objectif atteint" if atteint else "objectif non atteint"
    exemple = (" (exemple depuis theta1=90°, theta2=0°)" if est_bras(environnement) else "")
    ax.set_title(f"{nom} : meilleure action argmax Q(s, a) et chemin glouton{exemple}\n"
                 f"{len(chemin) - 1} déplacements, retour {retour:.2f}, {statut}")
    legende_cases(fig, environnement, "état jamais visité")
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graphe 6 : nombre de passages par chaque état
# ---------------------------------------------------------------------------
def graphe_visites(algorithme, environnement, nom, fichier):
    visites = en_grille(algorithme.visites, environnement)

    fig, ax = plt.subplots(figsize=taille_carte(environnement), layout="constrained")
    image = afficher_comptages(ax, visites, environnement, visites.max())

    fig.colorbar(image, ax=ax, shrink=0.8, format="%d", label="passages (échelle linéaire)")
    ax.set_title(f"{nom} : nombre de passages par état\n"
                 f"{int(visites.sum())} passages, "
                 f"{np.count_nonzero(visites)} états visités sur {visites.size}")
    legende_cases(fig, environnement, "jamais visité")
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graphe 7 : epsilon moyen de chaque état, en fonction de sa Q-value
# ---------------------------------------------------------------------------
def graphe_carte_epsilon(algorithme, environnement, nom, fichier):
    """Epsilon ne dépend que de l'épisode : la carte montre à quelle époque de
    l'entraînement chaque état a été visité, donc sous quel niveau d'exploration.
    """
    visites = algorithme.visites
    epsilon_moyen = np.full(environnement.nb_etats, np.nan)
    visitees = visites > 0
    epsilon_moyen[visitees] = algorithme.somme_epsilons[visitees] / visites[visitees]
    epsilon_moyen = en_grille(epsilon_moyen, environnement)
    valeurs = grille_valeurs(algorithme, environnement)

    fig, (ax_gauche, ax_droite) = plt.subplots(1, 2, figsize=(14, 6.5), layout="constrained")

    image = ax_gauche.imshow(epsilon_moyen, cmap=ORANGES, vmin=algorithme.epsilon_min,
                             vmax=max(algorithme.epsilons))
    preparer_grille(ax_gauche, environnement)
    marquer_cases(ax_gauche, environnement)
    fig.colorbar(image, ax=ax_gauche, shrink=0.8, label="epsilon moyen lors des passages")
    ax_gauche.set_title("Carte d'epsilon : clair = visité tard, foncé = visité tôt")

    if not est_bras(environnement):
        for (ligne, colonne), v in np.ndenumerate(epsilon_moyen):
            if np.isfinite(v):
                ecrire_valeur(ax_gauche, ligne, colonne, f"{v:.2f}", image.norm(v))

    # À droite : un point par état. Epsilon moyen de tout l'entraînement en pointillés.
    sur_chemin = np.zeros(environnement.nb_etats)
    sur_chemin[chemin_glouton(algorithme, environnement)[0]] = 1
    sur_chemin = en_grille(sur_chemin, environnement) > 0
    valides = np.isfinite(valeurs)
    for masque, couleur, label in ((~sur_chemin, COULEURS[1], "hors du chemin glouton"),
                                   (sur_chemin, COULEURS[0], "sur le chemin glouton")):
        ax_droite.scatter(valeurs[masque & valides], epsilon_moyen[masque & valides], s=36,
                          alpha=0.7, color=couleur, linewidths=0, label=label)
    ax_droite.axhline(np.mean(algorithme.epsilons), color=ENCRE_SECONDAIRE, linestyle="--",
                      linewidth=1, label="epsilon moyen de l'entraînement")
    ax_droite.set_xlabel("V(s) = max Q(s, a) en fin d'entraînement")
    ax_droite.set_ylabel("epsilon moyen lors des passages")
    ax_droite.set_title("Epsilon en fonction de la Q-value (un point par état)")
    ax_droite.legend()

    fig.suptitle(f"{nom} : epsilon vu par chaque état")
    legende_cases(fig, environnement, "jamais visité")
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graphe 8 : nombre de mises à jour de chaque action, une carte par action
# ---------------------------------------------------------------------------
def graphe_mises_a_jour(algorithme, environnement, nom, fichier):
    mises_a_jour = algorithme.mises_a_jour
    total = mises_a_jour.sum()

    taille = (12, 11) if est_bras(environnement) else (11, 10)
    fig, axes = plt.subplots(2, 2, figsize=taille, layout="constrained")

    for action, ax in enumerate(axes.flat):
        comptages = en_grille(mises_a_jour[:, action], environnement)
        image = afficher_comptages(ax, comptages, environnement, mises_a_jour.max())
        ax.set_title(f"{environnement.noms_actions[action]} : "
                     f"{comptages.sum():.0f} mises à jour "
                     f"({100 * comptages.sum() / total:.1f} %)")

    fig.colorbar(image, ax=axes, shrink=0.8, format="%d",
                 label="mises à jour de Q(s, a) (échelle linéaire)")
    fig.suptitle(f"{nom} : nombre de mises à jour par action ({total} au total)")
    legende_cases(fig, environnement, "action impossible ou jamais mise à jour")
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graphe 9 : lien entre les mises à jour et la stratégie epsilon-greedy
# ---------------------------------------------------------------------------
def barres_empilees(ax, noms, exploitations, explorations):
    positions = range(len(noms))
    ax.bar(positions, exploitations, color=COULEURS[0], edgecolor=FOND, linewidth=2,
           label="meilleure action (exploitation)")
    ax.bar(positions, explorations, bottom=exploitations, color=COULEURS[1], edgecolor=FOND,
           linewidth=2, label="tirage au hasard (exploration)")
    ax.set_xticks(positions, noms)
    ax.set_ylabel("mises à jour")


def graphe_lien_epsilon_greedy(algorithme, environnement, nom, fichier):
    episodes = range(1, len(algorithme.epsilons) + 1)
    explorations = algorithme.explorations
    exploitations = algorithme.mises_a_jour - explorations

    fig, (ax_temps, ax_actions, ax_rangs) = plt.subplots(1, 3, figsize=(15, 5.5),
                                                         layout="constrained")

    # À gauche : la part d'actions tirées au hasard suit epsilon.
    ax_temps.plot(episodes, algorithme.taux_exploration, color=COULEURS[1],
                   linewidth=0.8, alpha=0.65, label="part mesurée par épisode")
    ax_temps.plot(episodes, algorithme.epsilons, color=ENCRE_SECONDAIRE, linestyle="--",
                  linewidth=1, label="epsilon")
    ax_temps.set_xlabel("épisode")
    ax_temps.set_ylabel("part d'actions tirées au hasard")
    ax_temps.set_title("Le hasard suit epsilon")
    ax_temps.legend()

    # Au milieu : d'où viennent les mises à jour de chaque action.
    barres_empilees(ax_actions, environnement.noms_actions,
                    exploitations.sum(axis=0), explorations.sum(axis=0))
    ax_actions.set_title("Mises à jour par action")

    # À droite : dans chaque état, on classe les actions selon leur Q-value finale.
    rangs_exploitation = np.zeros(environnement.nb_actions)
    rangs_exploration = np.zeros(environnement.nb_actions)
    for etat in range(environnement.nb_etats):
        actions = environnement.actions_possibles(etat)
        classement = sorted(actions, key=lambda a: algorithme.Q[etat, a], reverse=True)
        for rang, action in enumerate(classement):
            rangs_exploitation[rang] += exploitations[etat, action]
            rangs_exploration[rang] += explorations[etat, action]

    noms_rangs = ["1re Q", "2e Q", "3e Q", "4e Q"]
    barres_empilees(ax_rangs, noms_rangs, rangs_exploitation, rangs_exploration)
    ax_rangs.set_title("Mises à jour selon le rang de l'action dans son état")

    fig.legend(*ax_actions.get_legend_handles_labels(), loc="outside lower center", ncols=2)
    fig.suptitle(f"{nom} : P(a | s) = epsilon / |A(s)| + (1 - epsilon) si a = argmax Q(s, .)")
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparaison labyrinthe / bras
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


def graphe_comparaison(agents, fichier):
    """agents : une entrée (nom, algorithme, environnement) par expérience."""
    noms = [nom for nom, _, _ in agents]
    mesures = [indicateurs(algorithme, environnement) for _, algorithme, environnement in agents]
    positions = range(len(agents))

    fig, (ax_visites, ax_couples, ax_reussite) = plt.subplots(1, 3, figsize=(15, 5.5),
                                                              layout="constrained")

    # À gauche : quelle part des passages se concentre sur les états les plus visités ?
    for nom, algorithme, environnement in agents:
        etats = environnement.nb_etats - len(getattr(environnement, "murs", []))
        visites = np.sort(algorithme.visites)[::-1][:etats]
        part_etats = 100 * np.arange(1, etats + 1) / etats
        ax_visites.plot(part_etats, 100 * np.cumsum(visites) / visites.sum(), label=nom)
    ax_visites.set_xlabel("% des états, du plus visité au moins visité")
    ax_visites.set_ylabel("% cumulé des passages")
    ax_visites.set_title("Concentration des passages")
    ax_visites.legend(loc="lower right")

    # Au milieu et à droite : une barre par expérience, valeur écrite au-dessus.
    panneaux = [
        (ax_couples, "mises_a_jour_par_couple", "Mises à jour par couple (s, a) autorisé",
         "{:.0f}"),
        (ax_reussite, "reussite_tous_departs", "Réussite de la politique depuis tous les états",
         "{:.1f} %"),
    ]
    for ax, cle, titre, format_valeur in panneaux:
        valeurs = [m[cle] for m in mesures]
        barres = ax.bar(positions, valeurs, color=COULEURS[:len(agents)], width=0.6)
        ax.bar_label(barres, [format_valeur.format(v) for v in valeurs], padding=4)
        ax.set_xticks(positions, noms, rotation=15, ha="right")
        ax.set_title(titre)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

    fig.suptitle("Labyrinthe et bras : même algorithme, espaces d'états différents")
    fig.savefig(fichier, dpi=150)
    plt.close(fig)
