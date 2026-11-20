"""50 entraînements Q-learning indépendants, puis réseau supervisé Keras.

Après CHAQUE épisode, on teste les 1294 départs de la Q-table en cours.
À la FIN de chaque entraînement, on conserve la Q-table finale et sa
moyenne finale. Parmi les 50 Q-tables finales, on choisit la plus petite
moyenne de mouvements (avec pénalité de 51 si un trajet échoue).

On n'utilise pas la meilleure Q-table intermédiaire d'un entraînement.
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
THETA1, THETA2 = 180, 90  # Position à tester après l'apprentissage.


# 1. Répéter le Q-learning 50 fois, en repartant de Q = 0 à chaque fois.
tables = []
moyennes = []
reussites_finales = []
historiques = []

for essai in range(NB_ESSAIS):
    # Graines différentes pour que les 50 entraînements soient indépendants.
    # creer_agent initialise une nouvelle Q-table et réinitialise epsilon.
    agent = creer_agent(
        mode="distance", nb_actions=9, episodes=NB_EPISODES,
        seed=42 + essai, entrainer=False
    )
    transitions = preparer_transitions(agent.env)
    evaluations = []

    def apres_episode(_mouvements, _reussi):
        # Après CHAQUE épisode, tester toutes les positions de départ.
        n_reussis, total, taux, cout = evaluer_globalement(agent, transitions)
        evaluations.append((len(evaluations) + 1, taux, cout))

    agent.apprendre(suivi_episodes=apres_episode)

    # IMPORTANT : garder la Q-table de FIN d'entraînement, pas un épisode
    # intermédiaire qui aurait eu temporairement un meilleur score.
    tables.append(agent.Q.copy())
    moyennes.append(evaluations[-1][2])
    reussites_finales.append(round(evaluations[-1][1] * len(agent.env.etats_depart) / 100))
    historiques.append(evaluations)

    print(f"Essai {essai + 1:02d}/{NB_ESSAIS} | "
          f"réussites {reussites_finales[-1]}/{len(agent.env.etats_depart)} | "
          f"moyenne finale {moyennes[-1]:.3f}")

# 2. Choisir la Q-table finale avec la plus petite moyenne.
meilleur = int(np.argmin(moyennes))
meilleure_q = tables[meilleur]
env = agent.env  # Les 50 essais utilisent exactement le même environnement.
print(f"\nMeilleur essai : {meilleur + 1}/{NB_ESSAIS}")
print(f"Réussites : {reussites_finales[meilleur]}/{len(env.etats_depart)}")
print(f"Plus petite moyenne finale : {moyennes[meilleur]:.3f}")

# Sauvegarder les 50 tables et leurs scores, ainsi que la table sélectionnée.
dossier = Path("experiences/q_tables")
dossier.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    dossier / "50_q_tables.npz",
    tables=np.array(tables),
    moyennes=np.array(moyennes),
    reussites=np.array(reussites_finales),
)
np.save(dossier / "meilleure_q_9.npy", meilleure_q)

# 3. Enseigner au réseau Keras les actions de la meilleure Q-table finale.
departs = np.array(env.etats_depart)
X = np.array([env.angles(int(s)) for s in departs], dtype="float32")
y = np.argmax(meilleure_q[departs], axis=1)

keras.utils.set_random_seed(42)
reseau = keras.Sequential([
    keras.Input(shape=(2,)),            # Entrée : (theta1, theta2).
    keras.layers.Rescaling(1 / 360),    # Normalisation interne.
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(9, activation="softmax"),
])
reseau.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
reseau.fit(X, y, epochs=NB_EPOQUES_RESEAU, batch_size=64, verbose=0)

# 4. Comparer la politique du réseau avec celle de la Q-table choisie.
politique = reseau.predict(X, verbose=0).argmax(axis=1)
print(f"\nAccord avec la meilleure Q-table : "
      f"{100 * np.mean(politique == y):.1f} %")

tous_angles = np.array([env.angles(s) for s in range(env.nb_etats)],
                       dtype="float32")
actions_reseau = reseau.predict(tous_angles, verbose=0).argmax(axis=1)
n_reussis, total, taux, cout = evaluer_globalement(
    agent, transitions, actions_politique=actions_reseau
)
print(f"Trajets du réseau : {n_reussis}/{total} réussis ({taux:.2f} %) | "
      f"moyenne pénalisée {cout:.3f}")

# 5. Exemple de vecteur 0/1 pour une position.
if (THETA1, THETA2) in [env.angles(s) for s in env.etats_objectif]:
    print("Le bras est déjà sur la cible.")
else:
    scores = reseau.predict(np.array([[THETA1, THETA2]], dtype="float32"),
                            verbose=0)[0]
    action = int(np.argmax(scores))
    vecteur = [0] * 9
    vecteur[action] = 1
    print(f"Position : ({THETA1}, {THETA2})")
    print("Vecteur du réseau :", vecteur)

# Afficher les deux graphiques de l'essai sélectionné,
# avec les valeurs calculées après chacun de ses 5000 épisodes.
tracer(historiques[meilleur], agent.max_pas, "distance", 9)
