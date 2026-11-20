"""Comparer deux réseaux Keras entraînés sur la même meilleure Q-table.

Réseau A : (theta1, theta2) -> 9 actions
Réseau B : (cos(theta1), sin(theta1), cos(theta2), sin(theta2)) -> 9 actions

Si la Q-table sauvegardée existe, on l'utilise directement.
Sinon, on la crée avec les 50 essais de Q-learning habituels.
"""

from pathlib import Path

import keras
import numpy as np

from experiences.animation_bras import creer_agent
from experiences.suivi_apprentissage import (
    evaluer_globalement, preparer_transitions, tracer
)

NB_ESSAIS = 50
NB_EPISODES = 5000
NB_EPOQUES_RESEAU = 200
THETA1, THETA2 = 180, 90

DOSSIER_Q = Path("experiences/q_tables")
CHEMIN_Q = DOSSIER_Q / "meilleure_q_9.npy"


# 1. Récupérer la meilleure Q-table, ou la générer si elle n'existe pas.
agent = creer_agent(mode="distance", nb_actions=9, episodes=NB_EPISODES,
                    seed=42, entrainer=False)
env = agent.env
transitions = preparer_transitions(env)
historique_meilleur = None

if CHEMIN_Q.exists():
    meilleure_q = np.load(CHEMIN_Q, allow_pickle=False)
    if meilleure_q.shape != agent.Q.shape or not np.isfinite(meilleure_q).all():
        raise ValueError("La Q-table sauvegardée n'a pas la bonne taille.")
    agent.Q = meilleure_q.copy()
    n, total, taux, cout = evaluer_globalement(agent, transitions)
    print(f"Q-table existante chargée : {n}/{total} réussites, "
          f"moyenne {cout:.3f}")
else:
    tables = []
    moyennes = []
    reussites = []

    for essai in range(NB_ESSAIS):
        # Chaque essai repart d'une Q-table nulle et d'un epsilon initial.
        agent = creer_agent(
            mode="distance", nb_actions=9, episodes=NB_EPISODES,
            seed=42 + essai, entrainer=False
        )
        evaluations = []

        def apres_episode(_mouvements, _reussi):
            # Tester TOUS les 1294 départs après chaque épisode.
            n, total, taux, cout = evaluer_globalement(agent, transitions)
            evaluations.append((len(evaluations) + 1, taux, cout))

        agent.apprendre(suivi_episodes=apres_episode)

        # Garder seulement la Q-table FINALE de cet entraînement.
        tables.append(agent.Q.copy())
        moyennes.append(evaluations[-1][2])
        reussites.append(round(evaluations[-1][1] * len(env.etats_depart) / 100))

        if essai == 0:
            historiques = []
        historiques.append(evaluations)

        print(f"Essai {essai + 1:02d}/{NB_ESSAIS} : "
              f"{reussites[-1]}/{len(env.etats_depart)} réussites | "
              f"moyenne finale {moyennes[-1]:.3f}")

    meilleur = int(np.argmin(moyennes))
    meilleure_q = tables[meilleur]
    historique_meilleur = historiques[meilleur]
    agent.Q = meilleure_q.copy()

    DOSSIER_Q.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DOSSIER_Q / "50_q_tables.npz",
        tables=np.array(tables),
        moyennes=np.array(moyennes),
        reussites=np.array(reussites),
    )
    np.save(CHEMIN_Q, meilleure_q)
    print(f"\nMeilleure Q-table : essai {meilleur + 1} "
          f"(moyenne finale {moyennes[meilleur]:.3f})")


# 2. Même liste d'états et mêmes actions attendues pour les deux réseaux.
departs = np.array(env.etats_depart)
angles_tous = np.array([env.angles(s) for s in range(env.nb_etats)],
                      dtype="float32")
angles = angles_tous[departs]
y = np.argmax(meilleure_q[departs], axis=1)


def transformer_angles(angles_entree):
    """Représentation alternative avec les sinus et cosinus."""
    r = np.deg2rad(angles_entree)
    return np.column_stack([
        np.cos(r[:, 0]), np.sin(r[:, 0]),
        np.cos(r[:, 1]), np.sin(r[:, 1]),
    ]).astype("float32")


# Réseau A : les 2 angles bruts (Keras les divisera par 360).
# Réseau B : 4 valeurs trigonométriques pour les mêmes 2 angles.
comparaisons = [
    ("angles", angles, angles_tous),
    ("trigo", transformer_angles(angles), transformer_angles(angles_tous)),
]

dossier_modeles = Path("experiences/modeles_reseau")
dossier_modeles.mkdir(parents=True, exist_ok=True)
resultats = []

# 3. Même étiquettes, deux réseaux distincts avec les mêmes couches cachées.
for nom, X, X_tous in comparaisons:
    keras.utils.set_random_seed(42)
    reseau = keras.Sequential()
    reseau.add(keras.Input(shape=(X.shape[1],)))

    if nom == "angles":
        reseau.add(keras.layers.Rescaling(1 / 360))

    reseau.add(keras.layers.Dense(64, activation="relu"))
    reseau.add(keras.layers.Dense(64, activation="relu"))
    reseau.add(keras.layers.Dense(9, activation="softmax"))
    reseau.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    reseau.fit(X, y, epochs=NB_EPOQUES_RESEAU, batch_size=64, verbose=0)

    # Accord avec les actions de la Q-table choisie.
    actions_apprises = reseau.predict(X, verbose=0).argmax(axis=1)
    accord = 100 * np.mean(actions_apprises == y)

    # Tester réellement les déplacements du réseau sur les 1294 départs.
    actions_tous = reseau.predict(X_tous, verbose=0).argmax(axis=1)
    n, total, taux, cout = evaluer_globalement(
        agent, transitions, actions_politique=actions_tous
    )
    resultats.append((nom, accord, taux, cout))
    reseau.save(dossier_modeles / f"bras_{nom}.keras")

    # Prédire un vecteur 0/1 pour la position souhaitée.
    if (THETA1, THETA2) not in [env.angles(s) for s in env.etats_objectif]:
        position = np.array([[THETA1, THETA2]], dtype="float32")
        if nom == "trigo":
            position = transformer_angles(position)
        action = int(np.argmax(reseau.predict(position, verbose=0)[0]))
        vecteur = [0] * 9
        vecteur[action] = 1
        print(f"Réseau {nom} : {accord:.1f}% d'accord avec Q | "
              f"{n}/{total} réussites | moyenne {cout:.3f}")
        print(f"  Position ({THETA1}, {THETA2}) -> {vecteur}")
    else:
        print(f"Réseau {nom} : {accord:.1f}% d'accord avec Q | "
              f"{n}/{total} réussites | moyenne {cout:.3f}")

# 4. Comparer les résultats sur la même Q-table et les mêmes 1294 départs.
print("\nComparaison des deux réseaux :")
print("Entrées             Accord Q    Réussite    Coût moyen pénalisé")
for nom, accord, taux, cout in resultats:
    print(f"{nom:<19} {accord:>6.1f}%     {taux:>6.2f}%    {cout:>8.3f}")

# Afficher les deux graphiques Q-learning si on vient d'entraîner les 50 Q-tables.
if historique_meilleur is not None:
    tracer(historique_meilleur, agent.max_pas, "distance", 9)
