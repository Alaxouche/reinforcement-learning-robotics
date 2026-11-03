# Graphiques des expériences Q-learning

Depuis la racine du dépôt :

```powershell
py -m pip install -r requirements.txt
py -m experiences.graphes
```

Le programme entraîne le **labyrinthe 5×5** et les trois modes du bras
(**naive, smart, distance**) avec la graine `42`. Il enregistre des PNG dans :

- `experiences/resultats/labyrinthes/`
- `experiences/resultats/bras_robotique/naive/`
- `experiences/resultats/bras_robotique/smart/`
- `experiences/resultats/bras_robotique/distance/`
- `experiences/resultats/comparaison_labyrinthe_bras.png`

Les anciens fichiers `1_convergence.png`, `2_epsilon_seeds.png`,
`5_cartes_q.png`, `5b_carte_valeurs.png`, `7_carte_epsilon.png`,
`8_mises_a_jour.png` et `9_lien_epsilon_greedy.png` sont supprimés
lors de la prochaine génération : ils ne font plus partie de l'analyse.

## Graphiques conservés

| Fichier | Interprétation |
|---|---|
| `3_cartes_q_brutes.png` | Q(s,a) brute pour les quatre actions, du bleu clair (faible) au bleu foncé (élevée) |
| `4_meilleure_action.png` | Politique gloutonne et chemin suivi : **flèches pour le labyrinthe**, **theta1+, theta1-, theta2+, theta2- pour le bras** |
| `6_visites.png` | Nombre de passages par état |
| `comparaison_labyrinthe_bras.png` | Concentration des visites, mises à jour par couple et taux de réussite |

Les graphes 3, 4 et 6 sont créés pour chaque environnement.
Le dernier est une comparaison globale.

## Comment interpréter les cartes ?

**Bras 36 × 36 :** les lignes correspondent aux 36 valeurs de `theta1`
et les colonnes aux 36 valeurs de `theta2`. Une case représente une
configuration angulaire, pas une position cartésienne de l'extrémité.

**Graphe 3 :** les Q-values conservent leur signe et leur valeur
numérique. Les quatre actions d'un même environnement partagent la même
échelle de bleu, du minimum observé au maximum observé : plus foncé
signifie Q-value plus élevée, **pas forcément positive**. Le gris représente
une action impossible ou jamais essayée. Les échelles peuvent différer
entre les modes naive, smart et distance ; il ne faut donc pas comparer
leurs teintes directement.

**Graphe 4 :** dans le labyrinthe, le départ unique est noté `S`
et les flèches montrent la meilleure action (haut, bas, gauche, droite).
Pour le bras, le nom écrit dans chaque case correspond à l'action ayant
la plus grande Q-value (`theta1+`, `theta1-`, `theta2+` ou `theta2-`).
**Aucun `S` n'est affiché sur les cartes du bras**, puisque chaque
épisode d'entraînement commence depuis un état aléatoire. Le chemin
superposé est uniquement **un exemple** depuis la configuration
`(90°, 0°)`, pas le parcours de tous les épisodes. Son trait est
interrompu visuellement quand un angle passe de 350° à 0°.

**Graphe 6 :** les couleurs utilisent une **échelle log**
pour rendre visibles les écarts de comptage entre les états peu et très
fréquentés. Le gris indique un état jamais visité. Les visites
comptent aussi l'état initial de chaque épisode.

## Limites d'interprétation

Les comparaisons reposent ici sur une seule graine (`42`) ; elles
permettent de décrire une expérience, mais ne prouvent pas la convergence
mathématique des Q-values ni la supériorité générale d'une récompense.

Les récompenses des modes naive, smart et distance ont des échelles
différentes ; privilégier le taux de réussite et le nombre d'actions
pour les comparer.

Le programme utilise Matplotlib avec le moteur `Agg` : les images sont
**enregistrées sur disque**, sans fenêtre graphique.
