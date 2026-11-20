"""Apprentissage supervise : imiter notre Q-table avec deux reseaux Keras.

On compare :
  - 2 entrees (theta1, theta2) et 4 entrees (sinus/cosinus) ;
  - choix argmax et tirage selon les probabilites softmax.

La Q-table est CHARGEE, jamais modifiee ni recalculee ici.
Les quatre graphiques sont enregistres dans experiences/resultats_reseau/.
"""

from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np

from experiences.animation_bras import creer_agent
from experiences.suivi_apprentissage import evaluer_globalement, preparer_transitions


# Parametres de l'experience : identiques pour les deux reseaux.
NB_EPOQUES_RESEAU = 200
TAILLE_BATCH = 64
NB_REPETITIONS = 30          # Repetitions des trajectoires avec tirage aleatoire.
THETA1, THETA2 = 180, 90    # Position choisie pour l'exemple final.

# Toujours retrouver les fichiers a partir du dossier de ce script.
DOSSIER = Path(__file__).resolve().parent
CHEMIN_Q = DOSSIER / "q_tables" / "meilleure_q_9.npy"
DOSSIER_MODELES = DOSSIER / "modeles_reseau"
DOSSIER_GRAPHIQUES = DOSSIER / "resultats_reseau"


# ---------------------------------------------------------------------------
# ETAPE 1 : charger la Q-table de reference, SANS lancer de Q-learning.
# ---------------------------------------------------------------------------
if not CHEMIN_Q.is_file():
    raise FileNotFoundError(
        f"Q-table introuvable : {CHEMIN_Q}\n"
        "Recupere meilleure_q_9.npy depuis GitHub. "
        "Le script ne regenerera jamais la Q-table."
    )

# On recree uniquement l'ENVIRONNEMENT du bras (ses etats et ses mouvements).
# Le parametre entrainer=False empeche tout apprentissage Q-learning.
agent = creer_agent(mode="distance", nb_actions=9, entrainer=False)
env = agent.env
transitions = preparer_transitions(env)

meilleure_q = np.load(CHEMIN_Q, allow_pickle=False)
if meilleure_q.shape != agent.Q.shape or not np.isfinite(meilleure_q).all():
    raise ValueError(f"Q-table invalide : taille attendue {agent.Q.shape}.")

# Charger en memoire une COPIE ; le fichier enregistre ne change pas.
agent.Q = meilleure_q.copy()
n_q, total_q, taux_q, cout_q = evaluer_globalement(agent, transitions)
print(f"Q-table de reference : {n_q}/{total_q} reussites "
      f"({taux_q:.2f} %) | cout moyen {cout_q:.3f}")


# ---------------------------------------------------------------------------
# ETAPE 2 : fabriquer les exemples supervises a partir de cette Q-table.
# ---------------------------------------------------------------------------
# 1294 departs : les deux positions qui atteignent deja la cible sont exclues.
departs = np.array(env.etats_depart)

# Pour chaque etat, retrouver les deux angles reels du bras, en degres.
angles_tous = np.array([env.angles(s) for s in range(env.nb_etats)],
                       dtype="float32")
angles = angles_tous[departs]

# Reponse attendue y : l'indice de la meilleure action selon la Q-table.
# On n'essaie PAS de predire les 9 valeurs Q(s,a), seulement argmax(Q[s]).
y = np.argmax(meilleure_q[departs], axis=1)


def transformer_angles(angles_entree):
    """Representer chaque couple d'angles par quatre sinus/cosinus."""
    radians = np.deg2rad(angles_entree)
    return np.column_stack([
        np.cos(radians[:, 0]), np.sin(radians[:, 0]),
        np.cos(radians[:, 1]), np.sin(radians[:, 1]),
    ]).astype("float32")


# Memes etats et memes bonnes actions, seule la representation change.
comparaisons = [
    ("angles", angles, angles_tous),                        # 2 entrees
    ("trigo", transformer_angles(angles),
     transformer_angles(angles_tous)),                     # 4 entrees
]


# ---------------------------------------------------------------------------
# ETAPE 3 : evaluation probabiliste (sans modifier les poids du reseau).
# ---------------------------------------------------------------------------
def tester_tirage(probabilites, repetitions=NB_REPETITIONS):
    """Tester tous les departs avec tirage d'une action a CHAQUE mouvement.

    La meme liste de 1294 departs est rejouee 30 fois.
    Si la cible est atteinte, on enregistre le nombre de mouvements.
    Sinon, le cout de ce depart est 51 (limite de 50 mouvements).
    """
    rng = np.random.default_rng(2026)  # Reproductibilite des tirages.
    table_transitions = np.asarray(transitions)
    cibles = np.array(env.etats_objectif)

    moyennes = []
    reussites_totales = 0

    for _ in range(repetitions):
        # On repart de TOUS les etats initiaux a chaque repetition.
        etats = np.array(env.etats_depart)
        couts = np.full(len(etats), agent.max_pas + 1)
        actifs = np.ones(len(etats), dtype=bool)

        for pas in range(1, agent.max_pas + 1):
            indices = np.flatnonzero(actifs)
            if len(indices) == 0:
                break  # Toutes les trajectoires ont atteint la cible.

            # Tirage pondere par les 9 probabilites SOFTMAX de l'etat courant.
            cumul = np.cumsum(probabilites[etats[indices]], axis=1)
            cumul[:, -1] = 1.0  # Eviter les petits ecarts d'arrondi.
            tirages = rng.random(len(indices))
            actions = np.argmax(tirages[:, None] < cumul, axis=1)

            # Appliquer l'action, puis recuperer le nouvel etat du bras.
            etats[indices] = table_transitions[etats[indices], actions]

            # On arrete definitivement une trajectoire ayant atteint la cible.
            arrives = np.isin(etats[indices], cibles)
            couts[indices[arrives]] = pas
            actifs[indices[arrives]] = False

        reussites_totales += np.sum(~actifs)
        moyennes.append(np.mean(couts))

    taux = 100 * reussites_totales / (len(env.etats_depart) * repetitions)
    # Cet ecart-type mesure la variabilite des 30 COUTS MOYENS.
    return taux, float(np.mean(moyennes)), float(np.std(moyennes))


# ---------------------------------------------------------------------------
# ETAPE 4 : entrainer les DEUX reseaux, puis tester les QUATRE methodes.
# ---------------------------------------------------------------------------
DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)
resultats = []
historiques = {}

for nom, X, X_tous in comparaisons:
    # Memes conditions d'entrainement pour les deux representations.
    keras.utils.set_random_seed(42)
    reseau = keras.Sequential()
    reseau.add(keras.Input(shape=(X.shape[1],)))

    # Pour les angles bruts : normaliser par 360 DANS le reseau.
    if nom == "angles":
        reseau.add(keras.layers.Rescaling(1 / 360))

    # Deux couches cachees et neuf probabilites d'actions en sortie.
    reseau.add(keras.layers.Dense(64, activation="relu"))
    reseau.add(keras.layers.Dense(64, activation="relu"))
    reseau.add(keras.layers.Dense(9, activation="softmax"))

    # Apprendre a predire l'INDICE de la meilleure action de la Q-table.
    reseau.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])

    # fit() ajuste les poids ; history conserve la precision et la perte
    # mesurees apres chacune des 200 epoques sur les exemples d'entrainement.
    history = reseau.fit(X, y, epochs=NB_EPOQUES_RESEAU,
                         batch_size=TAILLE_BATCH, verbose=0)
    historiques[nom] = history.history

    # predict() ne modifie PAS les poids : les deux tests sont faits
    # sur le MEME reseau entraine et ses MEMES probabilites softmax.
    probas_tous = reseau.predict(X_tous, verbose=0)
    accord = 100 * np.mean(np.argmax(probas_tous[departs], axis=1) == y)

    # METHODE 1 : toujours prendre la plus grande probabilite (argmax).
    actions_argmax = np.argmax(probas_tous, axis=1)
    _, _, taux_argmax, cout_argmax = evaluer_globalement(
        agent, transitions, actions_politique=actions_argmax
    )
    resultats.append({
        "nom": nom, "choix": "argmax",
        "taux": taux_argmax, "cout": cout_argmax, "ecart": 0.0
    })

    # METHODE 2 : tirer une action suivant la loi softmax de chaque etat.
    taux_tirage, cout_tirage, ecart_tirage = tester_tirage(probas_tous)
    resultats.append({
        "nom": nom, "choix": "tirage",
        "taux": taux_tirage, "cout": cout_tirage, "ecart": ecart_tirage
    })

    # Sauvegarder seulement les modeles neuronaux, jamais la Q-table.
    reseau.save(DOSSIER_MODELES / f"bras_{nom}.keras")
    print(f"\nReseau {nom} : accord avec Q = {accord:.1f} %")
    print(f"  Argmax : {taux_argmax:.2f} % reussite | cout {cout_argmax:.3f}")
    print(f"  Tirage : {taux_tirage:.2f} % reussite | cout {cout_tirage:.3f} "
          f"(ecart-type des moyennes = {ecart_tirage:.3f})")

    # Exemple concret : un etat donne un vecteur de 9 probabilites.
    # Pour chaque facon de choisir, on obtient un vecteur 0/1.
    if (THETA1, THETA2) not in [env.angles(s) for s in env.etats_objectif]:
        position = np.array([[THETA1, THETA2]], dtype="float32")
        if nom == "trigo":
            position = transformer_angles(position)
        probas = np.asarray(reseau.predict(position, verbose=0)[0], dtype=float)
        probas /= probas.sum()  # Probabilites somme = 1 pour le tirage.
        a_max = int(np.argmax(probas))
        a_tiree = int(np.random.default_rng(42).choice(9, p=probas))
        print(f"  Etat ({THETA1}, {THETA2}) | probabilites : "
              f"{np.round(probas, 3)}")
        for methode, action in [("argmax", a_max), ("tirage", a_tiree)]:
            vecteur = [0] * 9
            vecteur[action] = 1
            print(f"  {methode} -> {vecteur}")


# ---------------------------------------------------------------------------
# ETAPE 5 : afficher les quatre mesures de comparaison.
# ---------------------------------------------------------------------------
print("\nCOMPARAISON DES 4 METHODES")
print("Entrees   Choix     Reussite    Cout moyen    Ecart-type")
for r in resultats:
    ecart_texte = "-" if r["choix"] == "argmax" else f'{r["ecart"]:.3f}'
    print(f'{r["nom"]:<9} {r["choix"]:<9} {r["taux"]:>7.2f} %'
          f'    {r["cout"]:>8.3f}       {ecart_texte}')

meilleur = min(resultats, key=lambda r: r["cout"])
print(f"\nMeilleur reseau observe : {meilleur['nom']} + {meilleur['choix']}")
print(f"Q-table : {cout_q:.3f} mouvements moyens")
print(f"Meilleur reseau : {meilleur['cout']:.3f} mouvements moyens "
      f"(difference : {meilleur['cout'] - cout_q:+.3f})")
print("Ces tests utilisent les memes etats que l'entrainement : "
      "ils ne mesurent pas la generalisation hors de cette grille.")


# ---------------------------------------------------------------------------
# ETAPE 6 : tracer et sauvegarder les quatre graphiques du rapport.
# ---------------------------------------------------------------------------
DOSSIER_GRAPHIQUES.mkdir(parents=True, exist_ok=True)

# GRAPHE 1 : precision d'imitation de Q pendant les 200 epoques.
fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
for nom in historiques:
    epoques = np.arange(1, len(historiques[nom]["accuracy"]) + 1)
    ax.plot(epoques, np.array(historiques[nom]["accuracy"]) * 100, label=nom)
ax.set(title="Apprentissage : precision sur les exemples de la Q-table",
       xlabel="Epoque",
       ylabel="Actions predites correctement (%)")
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)
ax.legend()
fig.savefig(DOSSIER_GRAPHIQUES / "1_precision_entrainement.png", dpi=160)

# GRAPHE 2 : fonction de perte (erreur) sur les exemples d'entrainement.
fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
for nom in historiques:
    epoques = np.arange(1, len(historiques[nom]["loss"]) + 1)
    ax.plot(epoques, historiques[nom]["loss"], label=nom)
ax.set(title="Apprentissage : perte (entropie croisee)",
       xlabel="Epoque",
       ylabel="Perte sur les exemples d'entrainement")
ax.grid(alpha=0.3)
ax.legend()
fig.savefig(DOSSIER_GRAPHIQUES / "2_perte_entrainement.png", dpi=160)

# Memes quatre noms dans les graphes de performances.
noms = [f'{r["nom"]}\n{r["choix"]}' for r in resultats]
positions = np.arange(len(resultats))

# GRAPHE 3 : pourcentage des departs ayant atteint la cible.
fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
ax.bar(positions, [r["taux"] for r in resultats])
ax.axhline(taux_q, linestyle="--", color="black", label="Q-table de reference")
ax.set(title="Reussite sur les 1294 departs",
       ylabel="Taux de reussite (%)", xticks=positions, xticklabels=noms)
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.3)
ax.legend()
fig.savefig(DOSSIER_GRAPHIQUES / "3_taux_reussite.png", dpi=160)

# GRAPHE 4 : cout moyen, avec Q-table en pointille.
# Les barres d'erreur s'appliquent seulement au tirage et representent
# l'ecart-type des 30 moyennes obtenues (pas un intervalle de confiance).
fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
ax.bar(positions, [r["cout"] for r in resultats],
       yerr=[r["ecart"] for r in resultats], capsize=4)
ax.axhline(cout_q, linestyle="--", color="black", label="Q-table de reference")
ax.set(title="Cout moyen des trajectoires (echec = 51)",
       ylabel="Mouvements moyens penalises",
       xticks=positions, xticklabels=noms)
ax.grid(axis="y", alpha=0.3)
ax.legend()
fig.savefig(DOSSIER_GRAPHIQUES / "4_cout_moyen.png", dpi=160)

print(f"\n4 graphiques enregistres dans : {DOSSIER_GRAPHIQUES}")
plt.show()
