"""Anime le bras 2D en suivant EXCLUSIVEMENT la Q-table apprise.

Depuis la racine du dépôt :
    py -m experiences.animation_bras   # menu interactif : actions, reward, depart
    py -m experiences.animation_bras --mode naive --actions 4
    py -m experiences.animation_bras --gif
    py -m experiences.animation_bras --charger-q experiences/resultats/bras_robotique/q_distance_9.npy

Il s'agit d'un bras articulé *commandé* (cinématique), et non d'un
pendule physique libre : le modèle actuel ne contient ni gravité, ni inertie.
"""

import argparse
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import numpy as np

from experiences.entrainement_bras_robotique import MODES_BRAS
from src.algorithmes.q_learning import QLearning


def creer_agent(mode="distance", nb_actions=9, episodes=5000, seed=42,
                charger_q=None, enregistrer_q=None):
    """Entraîne un agent ou charge une Q-table déjà apprise (jamais une Q fictive)."""
    if episodes <= 0:
        raise ValueError("Le nombre d'épisodes doit être strictement positif.")
    np.random.seed(seed)
    env = MODES_BRAS[mode](depart_aleatoire=True, seed=seed,
                          nb_actions=nb_actions)
    agent = QLearning(environnement=env, alpha=0.2, gamma=0.95,
                      epsilon=1.0, epsilon_min=0.15, decroissance=0.9995,
                      nb_episodes=episodes, max_pas=50)
    if charger_q is not None:
        table = np.load(charger_q, allow_pickle=False)
        if table.shape != agent.Q.shape or not np.isfinite(table).all():
            raise ValueError(f"Q-table attendue : {agent.Q.shape}, "
                             f"reçue : {table.shape} (valeurs finies requises).")
        agent.Q = np.array(table, dtype=float, copy=True)
    else:
        agent.apprendre()
    if enregistrer_q is not None:
        destination = Path(enregistrer_q)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, agent.Q)
    return agent


def position_depart(env, type_depart="fixe", angles=None, seed=42):
    """Choisit un départ de l'animation, sans utiliser l'aléatoire d'entraînement."""
    if angles is not None:
        if type_depart == "aleatoire":
            raise ValueError("--angles et --depart aleatoire sont incompatibles.")
        i = env._indice_angle(env.angles_theta1, angles[0])
        j = env._indice_angle(env.angles_theta2, angles[1])
        s = env._indices_vers_etat(i, j)
    elif type_depart == "aleatoire":
        s = random.Random(seed + 10000).choice(env.etats_depart)
    else:
        s = env.etat_initial
    if s in env.etats_objectif:
        raise ValueError("Le bras est déjà sur la cible : choisir un autre départ.")
    return s


def trajectoire_q(agent, depart, max_pas=50):
    """Suit pi(s)=argmax_a Q(s,a) et renvoie états, actions, succès et cause.

    Aucune exploration et aucune mise à jour de Q pendant le rejeu.
    Une boucle est arrêtée immédiatement : la politique est déterministe.
    """
    if max_pas <= 0:
        raise ValueError("max_pas doit être strictement positif.")
    env = agent.env
    etat = env.reset(etat_depart=int(depart))
    etats = [etat]
    actions = []
    vus = {etat}
    if etat in env.etats_objectif:
        return etats, actions, True, "cible atteinte"
    for _ in range(max_pas):
        disponibles = env.actions_possibles(etat)
        if not disponibles:
            break
        action = agent.meilleure_action(etat)
        suivant, _, termine = env.step(action)
        actions.append(action)
        etats.append(suivant)
        if termine:
            return etats, actions, True, "cible atteinte"
        if suivant in vus:
            return etats, actions, False, "boucle de la politique"
        vus.add(suivant)
        etat = suivant
    return etats, actions, False, "limite de mouvements atteinte"


def angles_continus(env, etats, actions):
    """Déroule les angles à travers 0°/360° sans faire de rotation artificielle."""
    if len(etats) != len(actions) + 1:
        raise ValueError("Il faut un état de plus que d'actions.")
    poses = [tuple(map(float, env.angles(etats[0])))]
    for k, action in enumerate(actions):
        di, dj = env.deplacements[action]
        a, b = poses[-1]
        poses.append((a + di * env.pas_deg, b + dj * env.pas_deg))
        # Vérifie que chaque mouvement correspond aux états réels de l'environnement.
        if (int(poses[-1][0]) % 360, int(poses[-1][1]) % 360) != env.angles(etats[k + 1]):
            raise ValueError("Trajectoire Q incompatible avec les déplacements du bras.")
    return poses


def points_bras(longueurs, angles):
    """O, coude et extrémité ; theta2 est relatif à theta1."""
    theta1, theta2 = (math.radians(a) for a in angles)
    l1, l2 = longueurs
    x1, y1 = l1 * math.cos(theta1), l1 * math.sin(theta1)
    x2 = x1 + l2 * math.cos(theta1 + theta2)
    y2 = y1 + l2 * math.sin(theta1 + theta2)
    return (0.0, 0.0), (x1, y1), (x2, y2)


def frames_interpolees(poses, images_par_action=7, pause_finale=20):
    """Anime chaque incrément discret de 10° par une interpolation visuelle fluide."""
    if images_par_action < 2 or pause_finale < 0:
        raise ValueError("images_par_action >= 2 et pause_finale >= 0 requis.")
    frames = []
    for i in range(len(poses) - 1):
        a = np.asarray(poses[i], dtype=float)
        b = np.asarray(poses[i + 1], dtype=float)
        for sous_frame in range(images_par_action):
            t = sous_frame / images_par_action
            t = t * t * (3 - 2 * t)  # accélération/décélération douce
            frames.append((tuple(a + t * (b - a)), i))
    frames.extend([(poses[-1], len(poses) - 1)] * (pause_finale + 1))
    return frames


def animer(agent, etats, actions, reussi, raison, fps=24,
           images_par_action=7, gif=None, afficher=True):
    """Ouvre une animation Matplotlib ou sauvegarde un véritable GIF animé."""
    if fps <= 0:
        raise ValueError("fps doit être strictement positif.")
    env = agent.env
    poses = angles_continus(env, etats, actions)
    frames = frames_interpolees(poses, images_par_action, pause_finale=fps)
    longueurs = (env.l1, env.l2)
    limite = sum(longueurs) + 0.35
    points = [points_bras(longueurs, angles) for angles, _ in frames]
    trace_x = [p[2][0] for p in points]
    trace_y = [p[2][1] for p in points]

    fig, ax = plt.subplots(figsize=(8.8, 8.2))
    fig.patch.set_facecolor("#101722")
    ax.set_facecolor("#101722")
    ax.set_xlim(-limite, limite)
    ax.set_ylim(-limite, limite)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#425165", alpha=0.25, linestyle="--")
    ax.tick_params(colors="#b6c4d1")
    for spine in ax.spines.values():
        spine.set_color("#425165")
    ax.set_xlabel("x", color="#dce7f2")
    ax.set_ylabel("y", color="#dce7f2")
    mode = next((nom for nom, classe in MODES_BRAS.items()
                 if isinstance(env, classe)), "inconnue")
    ax.set_title(f"Bras à deux articulations — {env.nb_actions} actions — {mode}",
                 fontsize=14, color="white", pad=15)

    ax.add_patch(Circle(env.cible, env.tolerance, facecolor="#fb7185",
                        alpha=0.15, edgecolor="#fb7185", linewidth=2))
    ax.plot(*env.cible, marker="x", markersize=12, markeredgewidth=2.4,
            color="#fb7185", label="Cible")
    ax.plot(0, 0, marker="o", markersize=12, color="#e2e8f0", zorder=6)
    bras_1, = ax.plot([], [], color="#48d3e8", linewidth=10,
                      solid_capstyle="round", zorder=3, label="Segment 1")
    bras_2, = ax.plot([], [], color="#f4b865", linewidth=9,
                      solid_capstyle="round", zorder=4, label="Segment 2")
    coude, = ax.plot([], [], "o", markersize=12, color="white", zorder=7)
    effecteur, = ax.plot([], [], "o", markersize=10, color="#a5f3fc", zorder=8)
    trace, = ax.plot([], [], linewidth=1.8, color="#75c4db", alpha=0.75,
                     linestyle=":", label="Trajectoire de l'effecteur")
    infos = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top",
                    color="white", fontsize=11,
                    bbox=dict(facecolor="#172332", edgecolor="#425165",
                              boxstyle="round,pad=0.6", alpha=0.9))
    verdict = "CIBLE ATTEINTE" if reussi else f"ÉCHEC — {raison}"
    ax.text(0.02, 0.02, f"Politique gloutonne : {len(actions)} mouvements  |  {verdict}",
            transform=ax.transAxes, ha="left", color="#bfe8f0", fontsize=10)
    ax.legend(loc="lower right", facecolor="#1d2939", edgecolor="#455366",
              labelcolor="white")

    def afficher_frame(numero):
        o, a, m = points[numero]
        bras_1.set_data([o[0], a[0]], [o[1], a[1]])
        bras_2.set_data([a[0], m[0]], [a[1], m[1]])
        coude.set_data([a[0]], [a[1]])
        effecteur.set_data([m[0]], [m[1]])
        trace.set_data(trace_x[:numero + 1], trace_y[:numero + 1])
        theta1, theta2 = frames[numero][0]
        numero_action = min(frames[numero][1], len(actions))
        action_courante = (env.noms_actions[actions[numero_action]]
                           if numero_action < len(actions) else "fin")
        infos.set_text(f"Mouvement : {numero_action}/{len(actions)}\n"
                        f"θ1 = {theta1 % 360:.1f}°  |  θ2 = {theta2 % 360:.1f}°\n"
                        f"Action : {action_courante}\n"
                        f"Distance à la cible : {math.dist(m, env.cible):.3f}")
        return bras_1, bras_2, coude, effecteur, trace, infos

    animation = FuncAnimation(fig, afficher_frame, frames=len(frames),
                              interval=1000 / fps, blit=False,
                              repeat=bool(afficher))
    if gif is not None:
        path = Path(gif)
        path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(str(path), writer=PillowWriter(fps=fps))
        print(f"Animation GIF : {path}")
    if afficher:
        plt.show()
    plt.close(fig)
    return animation


def demander_choix(titre, options, defaut):
    """Demande un choix numéroté dans le terminal, avec validation."""
    while True:
        print(f"\n{titre}")
        for numero, (nom, _) in options.items():
            print(f"  {numero}. {nom}")
        try:
            saisie = input(f"Votre choix [{defaut}] : ").strip()
        except (EOFError, KeyboardInterrupt) as erreur:
            raise SystemExit("\nConfiguration annulée.") from erreur

        choix = saisie or defaut
        if choix in options:
            return options[choix][1]
        print("Choix invalide. Saisissez l'un des numéros affichés.")


def demander_angle(nom, valeur_defaut):
    """Lit un angle appartenant à la grille 0°, 10°, ..., 350°."""
    while True:
        try:
            texte = input(f"{nom} en degrés (0 à 350, multiple de 10) "
                          f"[{valeur_defaut}] : ").strip()
        except (EOFError, KeyboardInterrupt) as erreur:
            raise SystemExit("\nConfiguration annulée.") from erreur
        try:
            angle = int(texte) if texte else valeur_defaut
        except ValueError:
            print("Entrée invalide : il faut un nombre entier.")
            continue
        if 0 <= angle < 360 and angle % 10 == 0:
            return angle
        print("Angle invalide : choisir 0, 10, 20, ..., 350 degrés.")


def menu_interactif():
    """Choisit les paramètres du rejeu avant d'entraîner ou charger la Q-table.

    Le point de départ est un couple d'angles (theta1, theta2), car c'est
    la représentation des états utilisée par notre Q-learning.
    """
    print("\n" + "=" * 56)
    print(" SIMULATION DU BRAS ROBOTIQUE — PILOTAGE PAR Q-LEARNING")
    print("=" * 56)
    nb_actions = demander_choix(
        "1) Nombre d'actions du bras",
        {"1": ("4 actions (une articulation à la fois)", 4),
         "2": ("9 actions (déplacements simultanés possibles)", 9)},
        "2",
    )
    mode = demander_choix(
        "2) Fonction de récompense",
        {"1": ("Naive", "naive"),
         "2": ("Smart", "smart"),
         "3": ("Distance", "distance")},
        "3",
    )
    depart = demander_choix(
        "3) Position de départ du bras",
        {"1": ("Position habituelle (90°, 0°)", "fixe"),
         "2": ("Position aléatoire", "aleatoire"),
         "3": ("Choisir les angles theta1 et theta2", "personnalise")},
        "1",
    )

    angles = None
    if depart == "personnalise":
        env = MODES_BRAS[mode](nb_actions=nb_actions)
        while True:
            print("\nLa position du bras est définie par ses deux angles.")
            theta1 = demander_angle("Theta1 (segment 1)", 90)
            theta2 = demander_angle("Theta2 (relatif au segment 1)", 0)
            try:
                position_depart(env, angles=(theta1, theta2))
            except ValueError as erreur:
                print(f"Position impossible : {erreur}")
                print("Veuillez choisir une autre configuration.")
                continue
            angles = (theta1, theta2)
            break
        depart = "fixe"

    description_depart = ("aléatoire" if depart == "aleatoire"
                          else f"angles {angles}" if angles is not None
                          else "angles (90, 0)")
    print(f"\nConfiguration : {nb_actions} actions | récompense {mode} "
          f"| départ {description_depart}")
    return mode, nb_actions, depart, angles


def main():
    parser = argparse.ArgumentParser(
        description="Animer le bras contrôlé par la Q-table apprise (sans exploration).")
    parser.add_argument("--mode", choices=tuple(MODES_BRAS), default="distance")
    parser.add_argument("--actions", type=int, choices=(4, 9), default=9)
    parser.add_argument("--menu", action="store_true",
                        help="Afficher le menu de choix, même avec des arguments CLI.")
    parser.add_argument("--sans-menu", action="store_true",
                        help="Garder les valeurs CLI/par défaut sans demander de choix.")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depart", choices=("fixe", "aleatoire"), default="fixe")
    parser.add_argument("--angles", nargs=2, type=int, metavar=("THETA1", "THETA2"))
    parser.add_argument("--charger-q", type=Path)
    parser.add_argument("--enregistrer-q", type=Path)
    parser.add_argument("--gif", nargs="?", const="auto", default=None,
                        help="Enregistrer un GIF. Sans chemin, utilise experiences/resultats/bras_robotique/animations/.")
    parser.add_argument("--sans-fenetre", action="store_true",
                        help="Exporter sans ouvrir la fenêtre Matplotlib.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--images-par-action", type=int, default=7)
    args = parser.parse_args()
    if args.menu and args.sans_menu:
        parser.error("--menu et --sans-menu ne peuvent pas être combinés.")

    # Sans argument, un terminal interactif ouvre le menu automatiquement.
    # Avec des options CLI explicites, on conserve le comportement scriptable.
    interactif = args.menu or (
        len(sys.argv) == 1 and sys.stdin.isatty() and not args.sans_menu
    )
    if interactif:
        if args.angles is not None or args.depart != "fixe":
            parser.error("Avec --menu, choisissez le départ dans le menu "
                         "et n'utilisez ni --angles ni --depart.")
        args.mode, args.actions, args.depart, args.angles = menu_interactif()

    print("Q-table : chargement existant." if args.charger_q else
          f"Q-table : entraînement de {args.episodes} épisodes...")
    agent = creer_agent(args.mode, args.actions, args.episodes, args.seed,
                        args.charger_q, args.enregistrer_q)
    depart = position_depart(agent.env, args.depart, args.angles, args.seed)
    etats, actions, reussi, raison = trajectoire_q(agent, depart)
    x, y = agent.env.positions(depart)[-1]
    print(f"Départ : angles {agent.env.angles(depart)}, "
          f"extrémité M=({x:.3f}, {y:.3f})")
    print(f"Politique Q : {len(actions)} mouvements | {raison}")
    if args.gif == "auto":
        gif = (Path("experiences/resultats/bras_robotique/animations")
               / f"bras_{args.mode}_{args.actions}_actions.gif")
    else:
        gif = args.gif
    if args.sans_fenetre and not gif:
        parser.error("--sans-fenetre requiert --gif pour produire une animation.")
    animer(agent, etats, actions, reussi, raison,
           fps=args.fps, images_par_action=args.images_par_action,
           gif=gif, afficher=not args.sans_fenetre)


if __name__ == "__main__":
    main()
