"""Q-learning -> meilleure Q-table -> réseau supervisé Keras.

Après CHAQUE épisode, on évalue la Q-table sur tous les 1294 départs.
On conserve celle qui a le plus petit coût moyen pénalisé.
Le réseau Keras apprend ensuite à reproduire les actions de cette Q-table.
"""

from pathlib import Path

import keras
import numpy as np

from experiences.animation_bras import creer_agent
from experiences.suivi_apprentissage import (
    evaluer_globalement, preparer_transitions, tracer
)

NB_EPISODES = 5000
NB_EPOQUES_RESEAU = 200
THETA1, THETA2 = 180, 90  # Position de test.


def selectionner_q():
    """Entraîner notre Q-learning classique, sans réinitialiser Q entre épisodes."""
    agent = creer_agent(mode="distance", nb_actions=9,
                        episodes=NB_EPISODES, seed=42, entrainer=False)
    transitions = preparer_transitions(agent.env)

    meilleure_q = None
    meilleur_cout = float("inf")
    meilleures_reussites = -1
    meilleur_episode = 0
    numero_episode = 0
    mesures = []

    def apres_episode(_mouvements, _reussi):
        nonlocal meilleure_q, meilleur_cout, meilleures_reussites
        nonlocal meilleur_episode, numero_episode
        numero_episode += 1

        # Tester les 1294 positions de départ après CHAQUE épisode, sans epsilon.
        reussites, total, taux, cout = evaluer_globalement(agent, transitions)
        mesures.append((numero_episode, taux, cout))

        # Un trajet échoué en 50 pas reçoit 51 points de pénalité.
        # Garder la Q-table avec le coût moyen pénalisé le plus faible.
        if (cout < meilleur_cout - 1e-10 or
                (abs(cout - meilleur_cout) <= 1e-10
                 and reussites > meilleures_reussites)):
            meilleure_q = agent.Q.copy()
            meilleur_cout = cout
            meilleures_reussites = reussites
            meilleur_episode = numero_episode

        if numero_episode == 1 or numero_episode % 500 == 0:
            print(f"Épisode {numero_episode:>4} : "
                  f"{taux:6.2f} % réussite | coût moyen {cout:.3f}")

    agent.apprendre(suivi_episodes=apres_episode)
    return (agent, transitions, mesures, meilleure_q,
            meilleur_episode, meilleures_reussites, meilleur_cout)


# 1. Faire notre Q-learning et garder la meilleure Q-table observée.
agent, transitions, mesures, meilleure_q, episode, reussites, cout = selectionner_q()
env = agent.env
print(f"\nMeilleure Q-table obtenue à l'épisode {episode}")
print(f"Réussites : {reussites}/{len(env.etats_depart)}")
print(f"Coût moyen pénalisé : {cout:.3f} mouvements")

# Sauvegarder la Q-table choisie.
chemin_q = Path("experiences/q_tables/meilleure_q_9.npy")
chemin_q.parent.mkdir(parents=True, exist_ok=True)
np.save(chemin_q, meilleure_q)
print(f"Q-table sauvegardée : {chemin_q}")

# 2. Pour chaque état, la bonne réponse est argmax de la meilleure Q-table.
departs = np.array(env.etats_depart)
X = np.array([env.angles(int(s)) for s in departs], dtype="float32")
y = np.argmax(meilleure_q[departs], axis=1)

# 3. Apprentissage supervisé Keras : deux angles -> neuf actions.
keras.utils.set_random_seed(42)
reseau = keras.Sequential([
    keras.Input(shape=(2,)),
    keras.layers.Rescaling(1 / 360),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(9, activation="softmax"),
])
reseau.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])
reseau.fit(X, y, epochs=NB_EPOQUES_RESEAU, batch_size=64, verbose=0)

# Mesurer l'accord du réseau avec les actions de la Q-table choisie.
actions_predites = reseau.predict(X, verbose=0).argmax(axis=1)
print(f"\nAccord du réseau avec la Q-table : "
      f"{100 * np.mean(actions_predites == y):.1f} %")

# Tester les trajets du RÉSEAU sur tous les départs, comme pour Q-learning.
tous_angles = np.array([env.angles(s) for s in range(env.nb_etats)],
                       dtype="float32")
politique_reseau = reseau.predict(tous_angles, verbose=0).argmax(axis=1)
reussis_nn, total, taux_nn, cout_nn = evaluer_globalement(
    agent, transitions, actions_politique=politique_reseau
)
print(f"Trajets du réseau : {reussis_nn}/{total} réussis "
      f"({taux_nn:.2f} %) ; coût moyen pénalisé {cout_nn:.3f}")

# 4. Afficher un vecteur one-hot pour la position choisie.
if (THETA1, THETA2) in [env.angles(s) for s in env.etats_objectif]:
    print("Le bras est déjà sur la cible.")
else:
    scores = reseau.predict(np.array([[THETA1, THETA2]], dtype="float32"),
                            verbose=0)[0]
    action = int(np.argmax(scores))
    vecteur = [0] * 9
    vecteur[action] = 1
    print(f"État : ({THETA1}, {THETA2})")
    print("Action du réseau :", vecteur)

# Les deux graphiques habituels du Q-learning.
tracer(mesures, agent.max_pas, "distance", 9)
