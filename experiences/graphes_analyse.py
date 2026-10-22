"""Analyses visuelles complémentaires du Q-learning.

Ces tracés fonctionnent avec toute grille d'états bidimensionnelle prise en
charge : cases (labyrinthes) ou couples d'angles (bras robotique 2D).
Chaque carte bleue montre une seule action Q(s,a), jamais max_a Q(s,a).
"""

from pathlib import Path
import re
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch, Rectangle
import numpy as np


# Indices identiques à l'ordre des actions dans Labyrinthe et BrasRobotique2D.
FLECHES_LABYRINTHE = [(0, -1), (0, 1), (-1, 0), (1, 0)]
FLECHES_BRAS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def _geometrie(env):
    """Retourne dimensions, axes et indication de type pour une grille 2D."""
    if hasattr(env, "taille"):
        n = int(env.taille)
        return n, n, "Colonne", "Ligne", "labyrinthe"
    if hasattr(env, "angles_theta1") and hasattr(env, "angles_theta2"):
        return (len(env.angles_theta1), len(env.angles_theta2),
                "Theta2 (degres)", "Theta1 (degres)", "bras")
    raise ValueError("Visualisation spatiale disponible pour un labyrinthe ou un bras 2D.")


def _noms_actions(env):
    return list(getattr(env, "noms_actions", [f"Action {i}" for i in range(env.nb_actions)]))


def _identifiant(nom):
    nom = str(nom).replace("+", " plus ").replace("-", " moins ")
    texte = unicodedata.normalize("NFKD", nom)
    texte = "".join(c for c in texte if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", "_", texte).strip("_")


def _valeurs_actions(algorithme, env):
    """Crée quatre matrices Q, masquant les états/actions impossibles."""
    lignes, colonnes, _, _, _ = _geometrie(env)
    if lignes * colonnes != env.nb_etats:
        raise ValueError("La grille n'est pas compatible avec la numérotation des états.")
    if algorithme.Q.shape != (env.nb_etats, env.nb_actions):
        raise ValueError("Dimensions incompatibles entre la Q-table et l'environnement.")
    matrices = np.full((env.nb_actions, lignes, colonnes), np.nan, dtype=float)
    for etat in range(env.nb_etats):
        i, j = divmod(etat, colonnes)
        for action in env.actions_possibles(etat):
            matrices[action, i, j] = float(algorithme.Q[etat, action])
    return matrices


def _contexte_grille(ax, env, lignes, colonnes, genre):
    """Indique les murs/cibles, sans confondre états du bras et cases du labyrinthe."""
    if genre == "labyrinthe":
        for etat in getattr(env, "murs", []):
            i, j = divmod(etat, colonnes)
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                   color="dimgray", zorder=5))
        depart = getattr(env, "depart", None)
        arrivee = getattr(env, "arrivee", None)
        if depart is not None:
            i, j = divmod(depart, colonnes)
            ax.scatter(j, i, marker="s", s=80, color="#2aa858", edgecolor="white", zorder=8)
        if arrivee is not None:
            i, j = divmod(arrivee, colonnes)
            ax.scatter(j, i, marker="*", s=110, color="#ffb000", edgecolor="black", zorder=8)
    else:
        # Le point de départ et les états objectifs sont des configurations
        # angulaires, pas des positions (x,y) de l'effecteur.
        depart = env.etat_initial
        i, j = divmod(depart, colonnes)
        ax.scatter(j, i, marker="s", s=60, color="#2aa858", edgecolor="white", zorder=8)
        objectifs = getattr(env, "etats_objectif", ())
        if objectifs:
            x = [divmod(e, colonnes)[1] for e in objectifs]
            y = [divmod(e, colonnes)[0] for e in objectifs]
            ax.scatter(x, y, marker="*", s=42, color="#ffb000", edgecolor="black", zorder=8)


def _axes_grille(ax, env, lignes, colonnes, genre):
    ax.set_xlim(-.5, colonnes - .5)
    ax.set_ylim(lignes - .5, -.5)
    ax.set_aspect("equal")
    pas = max(1, int(np.ceil(max(lignes, colonnes) / 12)))
    xt = np.arange(0, colonnes, pas)
    yt = np.arange(0, lignes, pas)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    if genre == "bras":
        ax.set_xticklabels([env.angles_theta2[k] for k in xt])
        ax.set_yticklabels([env.angles_theta1[k] for k in yt])
    ax.set_xlabel("Colonne" if genre == "labyrinthe" else "Theta2 (degres)")
    ax.set_ylabel("Ligne" if genre == "labyrinthe" else "Theta1 (degres)")
    ax.set_xticks(np.arange(-.5, colonnes, 1), minor=True)
    ax.set_yticks(np.arange(-.5, lignes, 1), minor=True)
    if max(lignes, colonnes) <= 12:
        ax.grid(which="minor", color="white", linewidth=.7, alpha=.75)
    ax.tick_params(which="minor", bottom=False, left=False)


def _enregistrer(fig, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return destination


def moyenne_glissante(valeurs, fenetre):
    valeurs = np.asarray(valeurs, dtype=float)
    fenetre = max(1, int(fenetre))
    sorties = np.zeros_like(valeurs)
    for i in range(len(valeurs)):
        segment = valeurs[max(0, i + 1 - fenetre):i + 1]
        sorties[i] = np.nanmean(segment) if np.isfinite(segment).any() else np.nan
    return sorties


def graphe_convergence(algorithme, nom, destination):
    """Même triptyque que ton premier fichier : récompenses, pas, epsilon."""
    n = len(algorithme.recompenses)
    if n == 0:
        raise ValueError("Il faut lancer apprendre() avant les graphiques.")
    if len(algorithme.epsilons) != n:
        raise ValueError("L'historique epsilon est manquant : vérifier QLearning.")
    episodes = np.arange(1, n + 1)
    fenetre = max(10, n // 50)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))
    series = [
        (algorithme.recompenses, "Récompense cumulée"),
        (algorithme.nombre_actions, "Nombre d'actions"),
    ]
    for ax, (valeurs, etiquette) in zip(axes[:2], series):
        ax.plot(episodes, valeurs, alpha=.22, color="tab:blue", label="Par épisode")
        ax.plot(episodes, moyenne_glissante(valeurs, fenetre), linewidth=2,
                color="tab:orange", label=f"Moyenne glissante ({fenetre})")
        ax.set_ylabel(etiquette)
        ax.legend(loc="best")
        ax.grid(alpha=.2)
    axes[2].plot(episodes, algorithme.epsilons, color="tab:green")
    axes[2].set(xlabel="Épisode", ylabel="Epsilon (exploration)")
    axes[2].grid(alpha=.2)
    fig.suptitle(f"{nom} — entraînement du Q-learning")
    fig.tight_layout()
    return _enregistrer(fig, destination)


def graphe_surfaces_q(algorithme, env, nom, destination):
    """Q(s,a) en 3D par action ; barres pour cases, surfaces pour angles."""
    m = _valeurs_actions(algorithme, env)
    lignes, colonnes, nom_x, nom_y, genre = _geometrie(env)
    borne = max(1e-9, float(np.nanmax(np.abs(m))))
    normes = colors.Normalize(-borne, borne)
    noms = _noms_actions(env)
    fig = plt.figure(figsize=(13, 10))
    for action in range(env.nb_actions):
        ax = fig.add_subplot(2, 2, action + 1, projection="3d")
        v = m[action]
        yy, xx = np.where(np.isfinite(v))
        if genre == "labyrinthe":
            hauteurs = v[yy, xx]
            ax.bar3d(xx - .37, yy - .37, np.zeros(len(xx)),
                     .74, .74, hauteurs, shade=False,
                     color=plt.cm.RdBu(normes(hauteurs)))
        else:
            xx_grid, yy_grid = np.meshgrid(
                np.asarray(env.angles_theta2, dtype=float),
                np.asarray(env.angles_theta1, dtype=float))
            ax.plot_surface(xx_grid, yy_grid, np.ma.masked_invalid(v),
                            cmap="RdBu", vmin=-borne, vmax=borne,
                            linewidth=0, antialiased=False)
        ax.set(title=noms[action], xlabel=nom_x, ylabel=nom_y,
               zlabel="Q(s,a)", zlim=(-borne, borne))
        ax.view_init(elev=35, azim=-60)
    legende = plt.cm.ScalarMappable(norm=normes, cmap="RdBu")
    fig.colorbar(legende, ax=fig.axes, shrink=.6, pad=.08, label="Q(s,a) : rouge négatif, bleu positif")
    fig.suptitle(f"{nom} — Q-values en 3D, une vue par action")
    return _enregistrer(fig, destination)


def chemin_glouton(algorithme, environnement):
    """Politique gloutonne sans apprentissage ; stoppe aux boucles et au max_pas."""
    etat = environnement.reset()
    chemin = [int(etat)]
    visites = {int(etat)}
    retour = 0.
    atteint = False
    for _ in range(algorithme.max_pas):
        actions = environnement.actions_possibles(etat)
        if not actions:
            break
        action = algorithme.meilleure_action(etat)
        etat, recompense, termine = environnement.step(action)
        etat = int(etat)
        chemin.append(etat)
        retour += float(recompense)
        if termine:
            atteint = True
            break
        if etat in visites:
            break
        visites.add(etat)
    return chemin, retour, atteint


def graphe_meilleure_action(algorithme, env, nom, destination):
    """Carte de la politique gloutonne, chemin en surimpression."""
    lignes, colonnes, _, _, genre = _geometrie(env)
    chemin, retour, atteint = chemin_glouton(algorithme, env)
    fig, ax = plt.subplots(figsize=(10, 9) if genre == "bras" else (8, 7))
    politiques = np.full((lignes, colonnes), np.nan)
    for etat in range(env.nb_etats):
        actions = env.actions_possibles(etat)
        if actions:
            i, j = divmod(etat, colonnes)
            politiques[i, j] = algorithme.meilleure_action(etat)
    cmap = plt.get_cmap("Set2", env.nb_actions).copy()
    cmap.set_bad("#dddddd")
    ax.imshow(np.ma.masked_invalid(politiques), cmap=cmap,
              vmin=-.5, vmax=env.nb_actions-.5, alpha=.7)
    _axes_grille(ax, env, lignes, colonnes, genre)
    _contexte_grille(ax, env, lignes, colonnes, genre)
    fleches = FLECHES_LABYRINTHE if genre == "labyrinthe" else FLECHES_BRAS
    pas = 1 if genre == "labyrinthe" else max(1, int(np.ceil(lignes / 18)))
    for i in range(0, lignes, pas):
        for j in range(0, colonnes, pas):
            if np.isfinite(politiques[i, j]):
                dx, dy = fleches[int(politiques[i, j])]
                ax.arrow(j - .28*dx, i - .28*dy, .56*dx, .56*dy,
                         head_width=.16, length_includes_head=True,
                         color="#222222", zorder=6)
    for precedent, suivant in zip(chemin[:-1], chemin[1:]):
        i, j = divmod(precedent, colonnes)
        ni, nj = divmod(suivant, colonnes)
        # Ne trace pas de faux segments à travers la grille pour le modulo 360.
        if abs(ni-i) + abs(nj-j) == 1:
            ax.plot([j, nj], [i, ni], color="#ed7500", linewidth=3, zorder=7)
    i, j = divmod(chemin[-1], colonnes)
    ax.scatter(j, i, color="#ed7500", s=45, zorder=9)
    noms = _noms_actions(env)
    legende = [Patch(color=cmap(k), label=noms[k]) for k in range(env.nb_actions)]
    ax.legend(handles=legende, bbox_to_anchor=(1.02, 1), loc="upper left")
    statut = "réussi" if atteint else "non atteint"
    ax.set_title(f"{nom} — meilleure action par état\n"
                 f"Politique gloutonne : {len(chemin)-1} actions, retour {retour:.2f}, {statut}")
    return _enregistrer(fig, destination)


def graphes_cartes_q(algorithme, env, nom, dossier):
    """Une heatmap BLEUE par action (et PAS la valeur globale V(s))."""
    m = _valeurs_actions(algorithme, env)
    lignes, colonnes, _, _, genre = _geometrie(env)
    valides = m[np.isfinite(m)]
    if not len(valides):
        raise ValueError("Aucune action autorisée à représenter.")
    mini, maxi = float(valides.min()), float(valides.max())
    # Même échelle brute pour les quatre actions : comparaison honnête.
    if np.isclose(mini, maxi):
        marge = 1. if np.isclose(mini, 0) else abs(mini)*.05
        mini, maxi = mini-marge, maxi+marge
    noms = _noms_actions(env)
    dossier = Path(dossier)
    sorties = []
    for action in range(env.nb_actions):
        fig, ax = plt.subplots(figsize=(10, 8) if genre == "bras" else (7.2, 6))
        cmap = plt.cm.Blues.copy()
        cmap.set_bad("#d6d6d6")
        image = ax.imshow(np.ma.masked_invalid(m[action]), cmap=cmap,
                          vmin=mini, vmax=maxi, interpolation="nearest")
        _axes_grille(ax, env, lignes, colonnes, genre)
        _contexte_grille(ax, env, lignes, colonnes, genre)
        if max(lignes, colonnes) <= 12:
            for etat in range(env.nb_etats):
                i, j = divmod(etat, colonnes)
                valeur = m[action, i, j]
                if np.isfinite(valeur):
                    couleur = "white" if (valeur-mini)/(maxi-mini) > .6 else "black"
                    ax.text(j, i, f"{valeur:.2f}", ha="center", va="center",
                            fontsize=8, color=couleur)
        fig.colorbar(image, ax=ax, label="Q(s, action) — valeur brute")
        ax.set_title(f"{nom} — Q(s, {noms[action]})\n"
                     "Gris : action impossible ou état terminal", fontsize=12, pad=12)
        chemin = dossier / f"{13 + action:02d}_q_{_identifiant(noms[action])}.png"
        sorties.append(_enregistrer(fig, chemin))
    return sorties


def generer_graphes_analyse(algorithme, environnement, dossier, nom):
    """Génère les 7 nouveaux PNG d'analyse à la suite des 6 graphes génériques."""
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)
    fichiers = [
        graphe_convergence(algorithme, nom, dossier / "10_convergence_triptyque.png"),
        graphe_surfaces_q(algorithme, environnement, nom,
                          dossier / "11_surfaces_q_actions.png"),
        graphe_meilleure_action(algorithme, environnement, nom,
                               dossier / "12_meilleure_action.png"),
    ]
    fichiers.extend(graphes_cartes_q(algorithme, environnement, nom, dossier))
    return fichiers


def graphe_epsilon_seeds(resultats, nom, destination):
    """Compare plusieurs explorations et seeds sans prétendre prouver convergence.

    Chaque stratégie apporte des récompenses d'entraînement par seed et
    les récompenses de la politique gloutonne finale. Une boîte indique
    donc la dispersion des évaluations finales, pas un "épisode de convergence".
    """
    if not resultats:
        raise ValueError("Aucun résultat de comparaison.")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6))
    for res in resultats:
        courbes = np.asarray(res["recompenses"], dtype=float)
        nb_episodes = courbes.shape[1]
        fenetre = max(10, nb_episodes // 50)
        courbes_lisses = np.asarray([moyenne_glissante(c, fenetre) for c in courbes])
        moy = courbes_lisses.mean(axis=0)
        std = courbes_lisses.std(axis=0)
        ep = np.arange(1, nb_episodes+1)
        ligne, = ax.plot(ep, moy, label=res["label"])
        ax.fill_between(ep, moy-std, moy+std, color=ligne.get_color(), alpha=.18)
    ax.set(xlabel="Épisode", ylabel="Récompense cumulée (lissée)",
           title="Récompenses en entraînement : moyenne ± écart-type")
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    labels = [r["label"] for r in resultats]
    scores = [r["evaluations"] for r in resultats]
    bx.boxplot(scores, showfliers=False)
    for k, valeurs in enumerate(scores, 1):
        bx.scatter([k]*len(valeurs), valeurs, color="tab:blue", s=25, zorder=5)
    bx.set_xticks(np.arange(1, len(labels)+1), labels, rotation=25, ha="right")
    bx.set(ylabel="Récompense moyenne sans exploration",
           title="Dispersion des performances finales par seed")
    bx.grid(alpha=.2)
    fig.suptitle(f"{nom} — influence d'epsilon et de la seed")
    fig.tight_layout()
    return _enregistrer(fig, destination)
