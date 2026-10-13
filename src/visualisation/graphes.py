"""Fonctions de tracé des 5 graphes d'analyse du Q-learning (matplotlib)."""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # écrit des images sans ouvrir de fenêtre

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Flèche de chaque action (dx, dy) : 0=haut, 1=bas, 2=gauche, 3=droite.
# L'axe vertical est inversé (ligne 0 en haut), donc "haut" correspond à dy négatif.
FLECHES = [(0, -0.3), (0, 0.3), (-0.3, 0), (0.3, 0)]


# ---------------------------------------------------------------------------
# Petites fonctions utilitaires
# ---------------------------------------------------------------------------
def moyenne_glissante(valeurs, fenetre):
    """Moyenne des `fenetre` dernières valeurs, pour lisser une courbe."""
    resultat = []
    for i in range(len(valeurs)):
        debut = max(0, i + 1 - fenetre)
        resultat.append(np.mean(valeurs[debut:i + 1]))
    return resultat


def chemin_glouton(algorithme, environnement):
    """Suit toujours la meilleure action depuis le départ, sans exploration.

    Renvoie (chemin, retour, objectif_atteint). S'arrête si l'agent tourne en rond.
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
        deja_visite = nouvel_etat in chemin
        chemin.append(nouvel_etat)

        if termine:
            return chemin, retour, True
        if deja_visite:
            break

        etat = nouvel_etat

    return chemin, retour, False


def cases_dangereuses(environnement):
    """Liste des pièges (5x5) ou des feux (9x9). Vide pour le 3x3."""
    return getattr(environnement, "pieges", []) + getattr(environnement, "feux", [])


def valeur_etat(algorithme, environnement, etat):
    """V(s) = plus grande Q-value parmi les actions autorisées."""
    actions = environnement.actions_possibles(etat)
    if not actions:
        return 0.0
    return algorithme.Q[etat, actions].max()


def preparer_grille(ax, n):
    """Règle les axes pour afficher une grille n x n, ligne 0 en haut."""
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("colonne")
    ax.set_ylabel("ligne")
    ax.set_aspect("equal")


def colorier_case(ax, etat, n, couleur):
    """Remplit la case `etat` avec une couleur."""
    ligne, colonne = divmod(etat, n)
    ax.add_patch(Rectangle((colonne - 0.5, ligne - 0.5), 1, 1, color=couleur))


# ---------------------------------------------------------------------------
# Graphe 1 : convergence de l'apprentissage
# ---------------------------------------------------------------------------
def graphe_convergence(algorithme, nom, fichier):
    recompenses = algorithme.recompenses
    episodes = range(1, len(recompenses) + 1)
    fenetre = max(10, len(recompenses) // 50)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(episodes, recompenses, alpha=0.25, label="par épisode")
    axes[0].plot(episodes, moyenne_glissante(recompenses, fenetre),
                 label=f"moyenne glissante ({fenetre} épisodes)")
    axes[0].set_ylabel("récompense cumulée")
    axes[0].legend()

    axes[1].plot(episodes, algorithme.longueurs, alpha=0.25)
    axes[1].plot(episodes, moyenne_glissante(algorithme.longueurs, fenetre))
    axes[1].set_ylabel("nombre de pas")

    axes[2].plot(episodes, algorithme.epsilons, color="gray")
    axes[2].set_ylabel("epsilon")
    axes[2].set_xlabel("épisode")

    fig.suptitle(f"{nom} : convergence du Q-learning")
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graphe 2 : impact d'epsilon sur la répétabilité (plusieurs seeds)
# ---------------------------------------------------------------------------
def graphe_epsilon_seeds(resultats, nom, fichier):
    """resultats : une entrée par stratégie, avec
    "label", "recompenses" (une liste par seed) et "convergences" (un nombre par seed).
    """
    fig, (ax_gauche, ax_droite) = plt.subplots(1, 2, figsize=(13, 5))

    nb_episodes = len(resultats[0]["recompenses"][0])
    fenetre = max(10, nb_episodes // 50)
    episodes = np.arange(1, nb_episodes + 1)

    # À gauche : moyenne des seeds, et bande de +/- un écart-type.
    for res in resultats:
        courbes = [moyenne_glissante(r, fenetre) for r in res["recompenses"]]
        moyenne = np.mean(courbes, axis=0)
        ecart = np.std(courbes, axis=0)
        ligne, = ax_gauche.plot(episodes, moyenne, label=res["label"])
        ax_gauche.fill_between(episodes, moyenne - ecart, moyenne + ecart,
                               color=ligne.get_color(), alpha=0.15)

    ax_gauche.set_xlabel("épisode")
    ax_gauche.set_ylabel(f"récompense (moyenne glissante {fenetre})")
    ax_gauche.set_title(f"Moyenne ± écart-type sur {len(resultats[0]['recompenses'])} seeds")
    ax_gauche.legend()

    # À droite : une boîte par stratégie, et un point par seed.
    labels = [res["label"] for res in resultats]
    convergences = [res["convergences"] for res in resultats]
    ax_droite.boxplot(convergences, showfliers=False)
    for i, valeurs in enumerate(convergences, start=1):
        ax_droite.scatter([i] * len(valeurs), valeurs, s=20, zorder=3)
    ax_droite.set_xticks(range(1, len(labels) + 1))
    ax_droite.set_xticklabels(labels, rotation=20, ha="right")
    ax_droite.set_ylabel("épisode de convergence")
    ax_droite.set_title("Convergence de la politique gloutonne (un point par seed)")

    fig.suptitle(f"{nom} : impact d'epsilon sur la répétabilité")
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graphe 3 : Q(s, a) en 3D, une vue par action
# ---------------------------------------------------------------------------
def graphe_surfaces_q(algorithme, environnement, nom, fichier):
    n = environnement.taille
    borne = np.abs(algorithme.Q).max() or 1.0  # même échelle pour les 4 vues

    fig = plt.figure(figsize=(12, 10))

    for action in range(environnement.nb_actions):
        ax = fig.add_subplot(2, 2, action + 1, projection="3d")

        # Une barre par case où l'action est autorisée (pas de barre sur les murs).
        colonnes, lignes, hauteurs = [], [], []
        for etat in range(environnement.nb_etats):
            if action in environnement.actions_possibles(etat):
                ligne, colonne = divmod(etat, n)
                lignes.append(ligne)
                colonnes.append(colonne)
                hauteurs.append(algorithme.Q[etat, action])

        # Rouge = valeur négative, bleu = valeur positive.
        couleurs = plt.cm.RdBu((np.array(hauteurs) + borne) / (2 * borne))
        ax.bar3d(np.array(colonnes) - 0.4, np.array(lignes) - 0.4, 0,
                 0.8, 0.8, hauteurs, color=couleurs, shade=False)

        ax.set_title(f"action {action} : {environnement.noms_actions[action]}")
        ax.set_xlabel("colonne")
        ax.set_ylabel("ligne")
        ax.set_zlabel("Q(s, a)")
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_zlim(-borne, borne)
        ax.view_init(elev=42, azim=-60)

    fig.suptitle(f"{nom} : Q-values par action (murs et actions interdites absents)")
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graphe 4 : meilleure action dans chaque case et chemin glouton
# ---------------------------------------------------------------------------
def graphe_meilleure_action(algorithme, environnement, nom, fichier):
    n = environnement.taille
    chemin, retour, atteint = chemin_glouton(algorithme, environnement)

    fig, ax = plt.subplots(figsize=(0.8 * n + 3, 0.8 * n + 1.5))
    preparer_grille(ax, n)

    # Cases spéciales.
    for etat in environnement.murs:
        colorier_case(ax, etat, n, "dimgray")
    for etat in cases_dangereuses(environnement):
        colorier_case(ax, etat, n, "tomato")
    colorier_case(ax, environnement.arrivee, n, "limegreen")
    ligne, colonne = divmod(environnement.depart, n)
    ax.text(colonne, ligne, "S", ha="center", va="center", fontweight="bold")

    # Chemin glouton : on relie les centres des cases visitées.
    lignes = [divmod(e, n)[0] for e in chemin]
    colonnes = [divmod(e, n)[1] for e in chemin]
    ax.plot(colonnes, lignes, color="orange", linewidth=8, alpha=0.4, label="chemin glouton")

    # Une flèche par case : la meilleure action.
    for etat in range(environnement.nb_etats):
        if not environnement.actions_possibles(etat):
            continue  # mur ou arrivée
        ligne, colonne = divmod(etat, n)
        dx, dy = FLECHES[algorithme.meilleure_action(etat)]
        ax.arrow(colonne - dx, ligne - dy, 2 * dx, 2 * dy,
                 head_width=0.15, length_includes_head=True, color="black")

    statut = "objectif atteint" if atteint else "objectif non atteint"
    ax.set_title(f"{nom} : meilleure action argmax Q(s, a)\n"
                 f"{len(chemin) - 1} déplacements, retour {retour:.2f}, {statut}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Graphe 5 : carte couleur de V(s), normalisée entre 0 et 1
# ---------------------------------------------------------------------------
def graphe_carte_valeurs(algorithme, environnement, nom, fichier):
    n = environnement.taille
    dangers = cases_dangereuses(environnement)

    # Tableau n x n des valeurs. NaN (case vide) pour les murs.
    valeurs = np.full((n, n), np.nan)
    for etat in range(environnement.nb_etats):
        if etat not in environnement.murs:
            valeurs[divmod(etat, n)] = valeur_etat(algorithme, environnement, etat)

    # Normalisation : la pire case vaut 0, la meilleure vaut 1.
    mini = np.nanmin(valeurs)
    maxi = np.nanmax(valeurs)
    normalisees = (valeurs - mini) / (maxi - mini)

    fig, ax = plt.subplots(figsize=(0.8 * n + 3, 0.8 * n + 1.5))
    image = ax.imshow(normalisees, cmap="Blues", vmin=0, vmax=1)
    preparer_grille(ax, n)

    for etat in environnement.murs:
        colorier_case(ax, etat, n, "dimgray")

    # Valeur écrite dans chaque case.
    for etat in range(environnement.nb_etats):
        if etat in environnement.murs:
            continue
        ligne, colonne = divmod(etat, n)
        v = normalisees[ligne, colonne]
        texte = "G" if etat == environnement.arrivee else f"{v:.2f}"
        if etat in dangers:
            texte += "\ndanger"
        ax.text(colonne, ligne, texte, ha="center", va="center", fontsize=8,
                color="white" if v > 0.5 else "black")

    fig.colorbar(image, ax=ax, label="V(s) = max Q(s, a), normalisée")
    ax.set_title(f"{nom} : carte de valeur des états\n"
                 f"V brute entre {mini:.2f} et {maxi:.2f}")
    fig.tight_layout()
    fig.savefig(fichier, dpi=150)
    plt.close(fig)
