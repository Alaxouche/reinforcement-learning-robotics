"""Sept courbes d'apprentissage, independantes du type d'environnement."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _moyenne_glissante(valeurs, fenetre=50):
    v = np.asarray(valeurs, dtype=float)
    if len(v) == 0:
        return np.array([]), np.array([])
    fenetre = min(fenetre, len(v))
    valides = np.isfinite(v)
    cumul = np.cumsum(np.insert(np.where(valides, v, 0), 0, 0))
    quantites = np.cumsum(np.insert(valides.astype(int), 0, 0))
    total = cumul[fenetre:] - cumul[:-fenetre]
    nombre = quantites[fenetre:] - quantites[:-fenetre]
    moyenne = np.divide(total, nombre,
                        out=np.full(len(total), np.nan),
                        where=nombre > 0)
    return np.arange(fenetre, len(v) + 1), moyenne


def _enregistrer(fig, dossier, nom):
    chemin = dossier / nom
    fig.savefig(chemin, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return chemin


def _courbe(valeurs, titre, ordonnee, dossier, nom,
            moyenne=True, limites=None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(np.arange(1, len(valeurs) + 1), valeurs,
            alpha=0.45 if moyenne else 1, label="Par episode")
    if moyenne and len(valeurs) > 1:
        x, y = _moyenne_glissante(valeurs)
        ax.plot(x, y, linewidth=2, label="Moyenne glissante (50 episodes)")
        ax.legend()
    ax.set(title=titre, xlabel="Episode", ylabel=ordonnee)
    if limites is not None:
        ax.set_ylim(*limites)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _enregistrer(fig, dossier, nom)


def generer_graphiques(algorithme, dossier=None):
    """Genere 7 images PNG a partir des indicateurs de QLearning."""
    if not algorithme.recompenses:
        raise ValueError("Entrainer le Q-learning avant de tracer.")
    if dossier is None:
        dossier = Path(__file__).resolve().parent / "resultats"
    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)

    chemins = [
        _courbe(algorithme.recompenses,
                "Recompense cumulee par episode", "Recompense totale",
                dossier, "01_recompenses.png"),
        _courbe(algorithme.nombre_actions,
                "Nombre d'actions par episode", "Nombre d'actions",
                dossier, "02_nombre_actions.png"),
        _courbe(algorithme.variations_q,
                "Variation maximale de Q par episode", "Variation maximale",
                dossier, "03_variation_q.png", moyenne=False),
        _courbe(algorithme.couverture_etats,
                "Couverture cumulee des etats", "Etats visites (%)",
                dossier, "04_couverture_etats.png",
                moyenne=False, limites=(0, 100))
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(np.arange(1, len(algorithme.epsilons) + 1),
            algorithme.epsilons, linewidth=2,
            label="Epsilon (probabilite)")
    x, taux = _moyenne_glissante(algorithme.taux_exploration)
    ax.plot(x, taux, linewidth=2,
            label="Exploration observee (moyenne glissante)")
    ax.set(title="Exploration et exploitation",
           xlabel="Episode", ylabel="Proportion")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    chemins.append(
        _enregistrer(fig, dossier, "05_epsilon_exploration.png")
    )

    chemins.append(
        _courbe(algorithme.erreurs_td,
                "Erreur temporelle moyenne absolue", "Moyenne de |TD|",
                dossier, "06_erreur_td.png")
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(algorithme.episodes_evaluation,
            algorithme.recompenses_evaluation, marker="o")
    ax.set(title="Performance sans exploration",
           xlabel="Episodes d'entrainement",
           ylabel="Recompense moyenne en evaluation")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    chemins.append(
        _enregistrer(fig, dossier, "07_evaluation_sans_exploration.png")
    )
    return chemins
