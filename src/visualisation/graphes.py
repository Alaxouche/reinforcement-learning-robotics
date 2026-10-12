"""Graphes d'analyse du Q-learning sur les labyrinthes (matplotlib)."""

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle


# Palette (encre, surfaces, séries catégorielles, statuts).
SURFACE = "#fcfcfb"
ENCRE = "#0b0b0b"
ENCRE_SECONDAIRE = "#52514e"
ENCRE_DISCRETE = "#898781"
GRILLE = "#e1e0d9"
AXE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUR = "#383835"
FEU = "#d03b3b"
ARRIVEE = "#0ca30c"
CHEMIN = "#eb6834"

# Rampe séquentielle bleue (valeurs) et rampe divergente rouge <-> bleu (Q-values).
BLEUS = LinearSegmentedColormap.from_list(
    "bleus", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIVERGENTE = LinearSegmentedColormap.from_list(
    "rouge_bleu", ["#a32c2c", "#e34948", "#f0efec", "#3987e5", "#184f95"]
)

# Flèches des actions (dx, dy) en coordonnées d'image : 0=haut, 1=bas, 2=gauche, 3=droite.
FLECHES = [(0, -0.3), (0, 0.3), (-0.3, 0), (0.3, 0)]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXE,
    "axes.labelcolor": ENCRE_SECONDAIRE,
    "axes.titlecolor": ENCRE,
    "axes.grid": True,
    "grid.color": GRILLE,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": ENCRE_DISCRETE,
    "ytick.color": ENCRE_DISCRETE,
    "legend.frameon": False,
    "font.size": 10,
})


def moyenne_glissante(valeurs, fenetre):
    """Moyenne glissante (fenêtre tronquée au début de la série)."""
    valeurs = np.asarray(valeurs, dtype=float)
    cumul = np.cumsum(np.insert(valeurs, 0, 0.0))
    indices = np.arange(1, len(valeurs) + 1)
    debuts = np.maximum(0, indices - fenetre)

    return (cumul[indices] - cumul[debuts]) / (indices - debuts)


def chemin_glouton(algorithme, environnement):
    """Suit la politique gloutonne depuis le départ.

    Renvoie (chemin, retour, objectif_atteint). S'arrête en cas de boucle.
    """
    etat = environnement.reset()
    chemin = [etat]
    retour = 0.0

    for _ in range(algorithme.max_pas):
        if not environnement.actions_possibles(etat):
            break

        action = algorithme.meilleure_action(etat)
        nouvel_etat, recompense, termine = environnement.step(action)
        retour += recompense
        boucle = nouvel_etat in chemin
        chemin.append(nouvel_etat)

        if termine:
            return chemin, retour, True

        if boucle:
            break

        etat = nouvel_etat

    return chemin, retour, False


def _dangers(environnement):
    """Cases pénalisantes (pièges ou feux) selon le labyrinthe."""
    return list(getattr(environnement, "pieges", [])) + list(getattr(environnement, "feux", []))


def _grille_q(algorithme, environnement, action):
    """Q(s, action) sous forme de grille ; NaN pour les murs et actions interdites."""
    n = environnement.taille
    grille = np.full((n, n), np.nan)

    for etat in range(environnement.nb_etats):
        if etat == environnement.arrivee:
            grille[divmod(etat, n)] = 0.0
        elif action in environnement.actions_possibles(etat):
            grille[divmod(etat, n)] = algorithme.Q[etat, action]

    return grille


def _grille_valeur(algorithme, environnement):
    """V(s) = max_a Q(s, a) sur les actions autorisées ; NaN pour les murs."""
    n = environnement.taille
    grille = np.full((n, n), np.nan)

    for etat in range(environnement.nb_etats):
        if etat in environnement.murs:
            continue

        actions = environnement.actions_possibles(etat)
        valeur = algorithme.Q[etat, actions].max() if actions else 0.0
        grille[divmod(etat, n)] = valeur

    return grille


def _dessiner_cases(ax, environnement):
    """Murs, feux/pièges, départ et arrivée sur un axe en coordonnées de grille."""
    n = environnement.taille

    for etat in environnement.murs:
        ligne, colonne = divmod(etat, n)
        ax.add_patch(Rectangle((colonne - 0.5, ligne - 0.5), 1, 1, color=MUR, zorder=2))

    for etat in _dangers(environnement):
        ligne, colonne = divmod(etat, n)
        ax.add_patch(Rectangle((colonne - 0.5, ligne - 0.5), 1, 1,
                               color=FEU, alpha=0.85, zorder=2))
        ax.text(colonne, ligne + 0.32, "feu", ha="center", va="center",
                color="white", fontsize=7, zorder=5)

    ligne, colonne = divmod(environnement.arrivee, n)
    ax.add_patch(Rectangle((colonne - 0.5, ligne - 0.5), 1, 1, color=ARRIVEE, zorder=2))
    ax.text(colonne, ligne, "G", ha="center", va="center",
            color="white", fontweight="bold", zorder=5)

    ligne, colonne = divmod(environnement.depart, n)
    ax.text(colonne - 0.32, ligne - 0.32, "S", ha="center", va="center",
            color=ENCRE, fontweight="bold", fontsize=9, zorder=5)


def _cadre_grille(ax, n):
    """Axes d'une grille n x n : lignes vers le bas, quadrillage entre les cases."""
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color=AXE, linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("colonne")
    ax.set_ylabel("ligne")
    ax.set_aspect("equal")
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(True)


# ---------------------------------------------------------------------------
# Graphe 1 : convergence de l'apprentissage.
# ---------------------------------------------------------------------------
def graphe_convergence(algorithme, nom, fichier):
    recompenses = np.array(algorithme.recompenses)
    longueurs = np.array(algorithme.longueurs)
    episodes = np.arange(1, len(recompenses) + 1)
    fenetre = max(10, len(recompenses) // 50)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1.2]})

    ax = axes[0]
    ax.plot(episodes, recompenses, color=SERIES[0], alpha=0.25, linewidth=0.8,
            label="par épisode")
    ax.plot(episodes, moyenne_glissante(recompenses, fenetre), color=SERIES[0],
            linewidth=2, label=f"moyenne glissante ({fenetre} épisodes)")
    ax.set_ylabel("récompense cumulée")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(episodes, longueurs, color=SERIES[0], alpha=0.25, linewidth=0.8)
    ax.plot(episodes, moyenne_glissante(longueurs, fenetre), color=SERIES[0], linewidth=2)
    ax.set_ylabel("nombre de pas")

    ax = axes[2]
    ax.plot(episodes, algorithme.epsilons, color=ENCRE_SECONDAIRE, linewidth=2)
    ax.set_ylabel("epsilon")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("épisode")

    fig.suptitle(f"{nom} : convergence du Q-learning", color=ENCRE, fontsize=13)
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graphe 2 : impact d'epsilon sur la répétabilité (plusieurs seeds).
# ---------------------------------------------------------------------------
def graphe_epsilon_seeds(resultats, nom, fichier):
    """resultats : liste de dicts {label, recompenses (seeds x épisodes), convergences}."""
    fig, (ax_courbes, ax_boites) = plt.subplots(
        1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [2.2, 1]}
    )

    nb_seeds, nb_episodes = resultats[0]["recompenses"].shape
    fenetre = max(10, nb_episodes // 50)
    episodes = np.arange(1, nb_episodes + 1)

    for couleur, res in zip(SERIES, resultats):
        lissees = np.array([moyenne_glissante(r, fenetre) for r in res["recompenses"]])
        moyenne = lissees.mean(axis=0)
        ecart = lissees.std(axis=0)

        ax_courbes.fill_between(episodes, moyenne - ecart, moyenne + ecart,
                                color=couleur, alpha=0.15, linewidth=0)
        ax_courbes.plot(episodes, moyenne, color=couleur, linewidth=2, label=res["label"])

    ax_courbes.set_xlabel("épisode")
    ax_courbes.set_ylabel(f"récompense (moyenne glissante {fenetre})")
    ax_courbes.set_title(f"Apprentissage : moyenne ± écart-type sur {nb_seeds} seeds",
                         loc="left", fontsize=11)
    ax_courbes.legend(loc="lower right")

    convergences = [res["convergences"] for res in resultats]
    boites = ax_boites.boxplot(convergences, patch_artist=True, widths=0.55,
                               medianprops={"color": ENCRE, "linewidth": 1.5},
                               whiskerprops={"color": ENCRE_DISCRETE},
                               capprops={"color": ENCRE_DISCRETE},
                               showfliers=False)
    for couleur, boite in zip(SERIES, boites["boxes"]):
        boite.set_facecolor(couleur)
        boite.set_alpha(0.55)
        boite.set_edgecolor(couleur)

    rng = np.random.default_rng(0)
    for i, (couleur, valeurs) in enumerate(zip(SERIES, convergences), start=1):
        x = i + rng.uniform(-0.12, 0.12, size=len(valeurs))
        ax_boites.scatter(x, valeurs, s=22, color=couleur, edgecolor=SURFACE,
                          linewidth=1, zorder=3)

    ax_boites.set_xticks(range(1, len(resultats) + 1))
    ax_boites.set_xticklabels([res["label"] for res in resultats],
                              rotation=20, ha="right")
    ax_boites.set_ylabel("épisode de convergence")
    ax_boites.set_title("Convergence de la politique gloutonne\n(un point par seed)",
                        loc="left", fontsize=11)

    fig.suptitle(f"{nom} : impact d'epsilon sur la répétabilité (seeds différentes)",
                 color=ENCRE, fontsize=13)
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graphe 3 : surfaces Q(ligne, colonne) pour chaque action.
# ---------------------------------------------------------------------------
def graphe_surfaces_q(algorithme, environnement, nom, fichier):
    n = environnement.taille
    grilles = [_grille_q(algorithme, environnement, a) for a in range(environnement.nb_actions)]
    borne = max(np.nanmax(np.abs(g)) for g in grilles) or 1.0
    norme = TwoSlopeNorm(vmin=-borne, vcenter=0.0, vmax=borne)

    colonnes, lignes = np.meshgrid(np.arange(n), np.arange(n))
    fig = plt.figure(figsize=(12, 10))

    for action, grille in enumerate(grilles):
        ax = fig.add_subplot(2, 2, action + 1, projection="3d")
        ax.bar3d(colonnes[~np.isnan(grille)] - 0.4, lignes[~np.isnan(grille)] - 0.4,
                 np.zeros(np.count_nonzero(~np.isnan(grille))), 0.8, 0.8,
                 grille[~np.isnan(grille)],
                 color=DIVERGENTE(norme(grille[~np.isnan(grille)])),
                 edgecolor=SURFACE, linewidth=0.4, shade=False)
        ax.set_title(f"action {action} : {environnement.noms_actions[action]}",
                     color=ENCRE, fontsize=11)
        ax.set_xlabel("colonne")
        ax.set_ylabel("ligne")
        ax.set_zlabel("Q(s, a)")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_zlim(-borne, borne)
        ax.view_init(elev=42, azim=-60)
        for axe in (ax.xaxis, ax.yaxis, ax.zaxis):
            axe.set_pane_color(SURFACE)

    barre = fig.colorbar(plt.cm.ScalarMappable(norm=norme, cmap=DIVERGENTE),
                         ax=fig.axes, shrink=0.5, pad=0.06)
    barre.set_label("Q(s, a)")
    fig.suptitle(f"{nom} : Q-values par action (murs et actions interdites absents)",
                 color=ENCRE, fontsize=13)
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graphe 4 : meilleure action par état et chemin glouton.
# ---------------------------------------------------------------------------
def graphe_meilleure_action(algorithme, environnement, nom, fichier):
    n = environnement.taille
    chemin, retour, atteint = chemin_glouton(algorithme, environnement)

    fig, ax = plt.subplots(figsize=(0.8 * n + 3.5, 0.8 * n + 1.5))
    _cadre_grille(ax, n)
    _dessiner_cases(ax, environnement)

    lignes_chemin = [divmod(e, n)[0] for e in chemin]
    colonnes_chemin = [divmod(e, n)[1] for e in chemin]
    ax.plot(colonnes_chemin, lignes_chemin, color=CHEMIN, linewidth=8, alpha=0.35,
            solid_capstyle="round", zorder=3, label="chemin glouton")

    for etat in range(environnement.nb_etats):
        if etat in environnement.murs or etat == environnement.arrivee:
            continue
        if not environnement.actions_possibles(etat):
            continue

        ligne, colonne = divmod(etat, n)
        dx, dy = FLECHES[algorithme.meilleure_action(etat)]
        ax.annotate("", xy=(colonne + dx, ligne + dy),
                    xytext=(colonne - dx, ligne - dy),
                    arrowprops={"arrowstyle": "-|>", "color": ENCRE,
                                "linewidth": 1.4, "mutation_scale": 12},
                    zorder=4)

    statut = "objectif atteint" if atteint else "objectif non atteint"
    ax.set_title(f"{nom} : meilleure action argmax Q(s, a)\n"
                 f"{len(chemin) - 1} déplacements, retour {retour:.2f}, {statut}",
                 color=ENCRE, fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graphe 5 : carte couleur de la valeur V(s) normalisée entre 0 et 1.
# ---------------------------------------------------------------------------
def graphe_carte_valeurs(algorithme, environnement, nom, fichier):
    n = environnement.taille
    valeurs = _grille_valeur(algorithme, environnement)
    mini, maxi = np.nanmin(valeurs), np.nanmax(valeurs)
    normalisees = (valeurs - mini) / (maxi - mini) if maxi > mini else np.zeros_like(valeurs)
    dangers = _dangers(environnement)

    fig, ax = plt.subplots(figsize=(0.8 * n + 3.5, 0.8 * n + 1.5))
    image = ax.imshow(np.ma.masked_invalid(normalisees), cmap=BLEUS, vmin=0, vmax=1,
                      extent=(-0.5, n - 0.5, n - 0.5, -0.5))
    _cadre_grille(ax, n)

    for etat in environnement.murs:
        ligne, colonne = divmod(etat, n)
        ax.add_patch(Rectangle((colonne - 0.5, ligne - 0.5), 1, 1, color=MUR, zorder=2))

    for etat in dangers:
        ligne, colonne = divmod(etat, n)
        ax.add_patch(Rectangle((colonne - 0.46, ligne - 0.46), 0.92, 0.92, fill=False,
                               edgecolor=FEU, linewidth=2.5, zorder=3))

    for etat in range(environnement.nb_etats):
        if etat in environnement.murs:
            continue
        ligne, colonne = divmod(etat, n)
        v = normalisees[ligne, colonne]
        texte = "G" if etat == environnement.arrivee else f"{v:.2f}"
        if etat in dangers:
            texte += "\nfeu"
        ax.text(colonne, ligne, texte, ha="center", va="center", fontsize=8,
                color="white" if v > 0.4 else ENCRE, zorder=4)

    barre = fig.colorbar(image, ax=ax, ticks=[0, 0.5, 1], shrink=0.8)
    barre.set_label("V(s) = max Q(s, a), normalisée")
    barre.ax.set_yticklabels(["0 (pire)", "0.5", "1 (meilleur)"])
    ax.set_title(f"{nom} : carte de valeur des états\n"
                 f"V brute entre {mini:.2f} et {maxi:.2f}",
                 color=ENCRE, fontsize=12)
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
    plt.close(fig)
