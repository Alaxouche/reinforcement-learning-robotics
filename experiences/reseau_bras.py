"""Comparer 4 méthodes neuronales avec UNE Q-table sauvegardée.

Aucune génération ni modification de Q ici !
Le fichier meilleure_q_9.npy doit déjà exister.
"""

from pathlib import Path

import keras
import numpy as np

from experiences.animation_bras import creer_agent
from experiences.suivi_apprentissage import evaluer_globalement, preparer_transitions

NB_EPOQUES_RESEAU = 200
NB_REPETITIONS = 30
THETA1, THETA2 = 180, 90

# Le chemin est relatif à ce fichier : fonctionne depuis n'importe quel dossier.
CHEMIN_Q = Path(__file__).resolve().parent / "q_tables" / "meilleure_q_9.npy"


# 1. Charger notre Q-table définitive. Ne JAMAIS relancer le Q-learning.
if not CHEMIN_Q.is_file():
    raise FileNotFoundError(
        f"Q-table introuvable : {CHEMIN_Q}\n"
        "Place ta Q-table enregistrée (meilleure_q_9.npy) dans ce dossier.\n"
        "Le réseau ne régénère jamais la Q-table."
    )

agent = creer_agent(mode="distance", nb_actions=9, entrainer=False)
env = agent.env
transitions = preparer_transitions(env)

meilleure_q = np.load(CHEMIN_Q, allow_pickle=False)
if meilleure_q.shape != agent.Q.shape or not np.isfinite(meilleure_q).all():
    raise ValueError(f"Q-table invalide : attendu {agent.Q.shape} valeurs finies.")

agent.Q = meilleure_q.copy()
n, total, taux, cout = evaluer_globalement(agent, transitions)
print(f"Q-table sauvegardée chargée : {CHEMIN_Q.name}")
print(f"Réussites : {n}/{total} | coût moyen pénalisé : {cout:.3f}")

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

# 3. Évaluer un choix ALÉATOIRE selon les probabilités du réseau.
def tester_tirage(probabilites, repetitions=NB_REPETITIONS):
    """Rejouer tous les départs, plusieurs fois car les actions sont aléatoires."""
    rng = np.random.default_rng(2026)
    table_transitions = np.asarray(transitions)
    cibles = np.array(env.etats_objectif)
    moyennes = []
    total_reussites = 0

    for _ in range(repetitions):
        etats = np.array(env.etats_depart)
        couts = np.full(len(etats), agent.max_pas + 1)
        actifs = np.ones(len(etats), dtype=bool)

        for pas in range(1, agent.max_pas + 1):
            indices = np.flatnonzero(actifs)
            if len(indices) == 0:
                break

            # À CHAQUE mouvement, on tire une action suivant softmax.
            cumul = np.cumsum(probabilites[etats[indices]], axis=1)
            cumul[:, -1] = 1.0
            tirages = rng.random(len(indices))
            actions = np.argmax(tirages[:, None] < cumul, axis=1)
            etats[indices] = table_transitions[etats[indices], actions]

            arrives = np.isin(etats[indices], cibles)
            couts[indices[arrives]] = pas
            actifs[indices[arrives]] = False

        total_reussites += np.sum(~actifs)
        moyennes.append(np.mean(couts))

    taux = 100 * total_reussites / (len(env.etats_depart) * repetitions)
    return taux, float(np.mean(moyennes)), float(np.std(moyennes))


# 4. Entraîner les 2 réseaux, puis tester les 2 choix sur chacun.
print("\nComparaison : entrées (2 ou 4) × choix (argmax ou tirage)")
n_q, total, taux_q, cout_q = evaluer_globalement(agent, transitions)
print(f"Q-table de référence : {n_q}/{total} réussites | "
      f"moyenne pénalisée {cout_q:.3f}")

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

    # Dans les deux méthodes, le réseau renvoie les MÊMES probabilités.
    probas_tous = reseau.predict(X_tous, verbose=0)
    accord = 100 * np.mean(np.argmax(probas_tous[departs], axis=1) == y)

    # Méthode 1 : action la plus probable, toujours la même pour un état.
    actions_argmax = np.argmax(probas_tous, axis=1)
    n, total, taux_argmax, cout_argmax = evaluer_globalement(
        agent, transitions, actions_politique=actions_argmax
    )
    resultats.append((nom, "argmax", taux_argmax, cout_argmax, None))

    # Méthode 2 : tirage selon les probabilités, renouvelé à chaque mouvement.
    taux_tirage, cout_tirage, ecart = tester_tirage(probas_tous)
    resultats.append((nom, "tirage", taux_tirage, cout_tirage, ecart))

    reseau.save(dossier_modeles / f"bras_{nom}.keras")

    print(f"\nRéseau {nom} : accord avec Q = {accord:.1f}%")
    print(f"  Argmax : {taux_argmax:.2f}% réussites | moyenne {cout_argmax:.3f}")
    print(f"  Tirage : {taux_tirage:.2f}% réussites | "
          f"moyenne {cout_tirage:.3f} (écart-type {ecart:.3f}, "
          f"{NB_REPETITIONS} répétitions)")

    # Un exemple pour les DEUX façons de choisir une action.
    if (THETA1, THETA2) not in [env.angles(s) for s in env.etats_objectif]:
        position = np.array([[THETA1, THETA2]], dtype="float32")
        if nom == "trigo":
            position = transformer_angles(position)
        probas = reseau.predict(position, verbose=0)[0].astype(float)
        probas = probas / probas.sum()

        action_max = int(np.argmax(probas))
        action_tiree = int(np.random.default_rng(42).choice(9, p=probas))
        print(f"  État ({THETA1}, {THETA2})")
        print(f"  Probabilités : {np.round(probas, 3)}")
        for methode, action in [("argmax", action_max), ("tirage", action_tiree)]:
            vecteur = [0] * 9
            vecteur[action] = 1
            print(f"  {methode} -> {vecteur} | action {action}")

# 5. Tableau récapitulatif des 4 méthodes sur les mêmes départs.
print("\nCOMPARAISON DES 4 MÉTHODES")
print("Entrée   Choix     Réussites     Coût moyen   Écart-type")
for nom, methode, taux, cout, ecart in resultats:
    ecart_texte = "-" if ecart is None else f"{ecart:.3f}"
    print(f"{nom:<8} {methode:<9} {taux:>7.2f}%      "
          f"{cout:>8.3f}     {ecart_texte}")
