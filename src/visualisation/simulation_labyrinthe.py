import tkinter as tk


def lancer_simulation(lab, qlearning):
    """Ouvre une fenêtre qui anime les décisions de l'agent déjà construit.

    Paramètres :
        lab : objet Labyrinthe à dessiner ;
        qlearning : objet QLearning à entraîner et à utiliser.
    Retour : aucun ; la fenêtre Tkinter reste ouverte jusqu'à sa fermeture.
    Les petites fonctions ci-dessous sont internes à cette visualisation.
    """
    TAILLE_FENETRE = 450  # en pixels, les cases s'adaptent a la taille du labyrinthe
    TAILLE_CASE = TAILLE_FENETRE // lab.taille
    DELAI = 400  # millisecondes entre deux pas

    # Avant de montrer le robot, on entraîne la table Q.
    qlearning.apprendre()

    etat = lab.depart  # Position affichée à l'écran (variable de cette fonction).
    nb_pas = 0
    score = 0.0
    en_marche = False
    message = ""  # Texte affiché quand l'épisode est fini.

    def dessiner():
        """Efface et redessine la grille, le robot et le score. Retour : aucun."""
        canevas.delete("all")

        # Chaque case est dessinée à la position (ligne, colonne) correspondante.
        for case in range(lab.nb_etats):
            ligne, colonne = divmod(case, lab.taille)
            x, y = colonne * TAILLE_CASE, ligne * TAILLE_CASE

            if case == lab.depart:
                couleur = "#c8e6c9"
            elif case == lab.arrivee:
                couleur = "#ffe082"
            elif case in lab.murs:
                couleur = "#424242"
            elif case in lab.feux:
                couleur = "#ef5350"
            else:
                couleur = "white"

            canevas.create_rectangle(x, y, x + TAILLE_CASE, y + TAILLE_CASE,
                                     fill=couleur, outline="gray")
            canevas.create_text(x + 14, y + 14, text=str(case), fill="gray")

        # Après la grille, on dessine le robot au centre de sa case actuelle.
        ligne, colonne = divmod(etat, lab.taille)
        x = colonne * TAILLE_CASE + TAILLE_CASE / 2
        y = ligne * TAILLE_CASE + TAILLE_CASE / 2
        rayon = TAILLE_CASE / 4
        canevas.create_oval(x - rayon, y - rayon, x + rayon, y + rayon, fill="#1976d2")

        info.set("état : {}    pas : {}    score : {:.1f}    {}"
                 .format(etat, nb_pas, score, message))

    def episode_fini():
        """Dit si le robot ne peut plus avancer. Retour : bool."""
        return lab.est_terminal(etat) or nb_pas >= qlearning.max_pas

    def avancer():
        """Effectue UNE action apprise et programme le prochain pas si nécessaire."""
        # nonlocal : ces variables viennent de lancer_simulation(), pas d'avancer().
        nonlocal etat, nb_pas, score, en_marche, message

        if not en_marche:
            return

        # On exécute la meilleure action de l'agent dans le labyrinthe.
        etat = lab.etat_suivant(etat, qlearning.meilleure_action(etat))
        nb_pas += 1
        score += lab.recompense(etat)

        # On s'arrete a l'arrivee ou si l'agent tourne en rond. Un feu ne
        # termine pas l'episode : il coute seulement tres cher.
        if etat == lab.arrivee:
            message = "arrivée !"
        elif etat in lab.feux:
            message = "feu traversé : -10"
        elif nb_pas >= qlearning.max_pas:
            message = "trop de pas"
        else:
            message = ""

        en_marche = not episode_fini()
        dessiner()

        if en_marche:
            fenetre.after(DELAI, avancer)  # Rappelle avancer() après DELAI ms.

    def demarrer():
        """Démarre l'animation quand on clique sur le bouton Démarrer."""
        nonlocal en_marche

        # Sans ce test, un clic sur un feu ferait planter meilleure_action.
        if not en_marche and not episode_fini():
            en_marche = True
            avancer()

    def recommencer():
        """Replace le robot au départ et remet le compteur et le score à zéro."""
        nonlocal etat, nb_pas, score, en_marche, message

        en_marche = False
        etat, nb_pas, score, message = lab.depart, 0, 0.0, ""
        dessiner()

    # Construction des éléments visibles de la fenêtre.
    fenetre = tk.Tk()
    fenetre.title("Q-learning labyrinthe")

    cote = lab.taille * TAILLE_CASE
    canevas = tk.Canvas(fenetre, width=cote, height=cote)
    canevas.pack()

    info = tk.StringVar()
    tk.Label(fenetre, textvariable=info).pack(pady=5)

    tk.Button(fenetre, text="Démarrer", command=demarrer).pack(side="left", padx=20, pady=10)
    tk.Button(fenetre, text="Recommencer", command=recommencer).pack(side="right", padx=20, pady=10)

    dessiner()
    fenetre.mainloop()  # Attend les clics de l'utilisateur jusqu'à fermeture.
