import tkinter as tk

from apprentissage import QLearning
from labyrinthe import creer_labyrinthe

# Pour changer de labyrinthe, il suffit de changer ces deux lignes
lab = creer_labyrinthe("murs_feux_9x9")  # ou "murs_5x5", "murs_feux_9x9" 
qlearning = QLearning(lab)

TAILLE_FENETRE = 450  # en pixels, les cases s'adaptent a la taille du labyrinthe
TAILLE_CASE = TAILLE_FENETRE // lab.taille
DELAI = 400  # millisecondes entre deux pas

qlearning.apprendre()
qlearning.afficher_q()

etat = lab.depart
nb_pas = 0
score = 0.0
en_marche = False


def dessiner():
    canevas.delete("all")

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

    ligne, colonne = divmod(etat, lab.taille)
    x = colonne * TAILLE_CASE + TAILLE_CASE / 2
    y = ligne * TAILLE_CASE + TAILLE_CASE / 2
    rayon = TAILLE_CASE / 4
    canevas.create_oval(x - rayon, y - rayon, x + rayon, y + rayon, fill="#1976d2")

    info.set("état : {}    pas : {}    score : {:.1f}".format(etat, nb_pas, score))


def avancer():
    global etat, nb_pas, score, en_marche

    if not en_marche:
        return

    etat = lab.etat_suivant(etat, qlearning.meilleure_action(etat))
    nb_pas += 1
    score += lab.recompense(etat)
    dessiner()

    # On s'arrete a l'arrivee, ou si l'agent tourne en rond trop longtemps
    if etat == lab.arrivee or nb_pas >= qlearning.max_pas:
        en_marche = False
    else:
        fenetre.after(DELAI, avancer)


def demarrer():
    global en_marche

    if not en_marche and etat != lab.arrivee:
        en_marche = True
        avancer()


def recommencer():
    global etat, nb_pas, score, en_marche

    en_marche = False
    etat, nb_pas, score = lab.depart, 0, 0.0
    dessiner()


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
fenetre.mainloop()
