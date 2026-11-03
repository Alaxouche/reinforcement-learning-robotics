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
| `4_meilleure_action.png` | Meilleure action : flèches et chemin pour le labyrinthe, quatre couleurs pour les actions du bras (sans chemin superposé) |
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

**Graphe 4 :** dans le labyrinthe, le départ unique est noté `S`,
les flèches montrent la meilleure action (haut, bas, gauche, droite), et
le chemin glouton est superposé.

Pour le **bras**, aucune étiquette textuelle ni aucun chemin de référence
n'est superposé sur la grille 36 × 36. Chaque case visitée et non terminale
est colorée suivant l'action de plus grande Q-value :
- **bleu** : `theta1+` ;
- **orange** : `theta1-` ;
- **vert** : `theta2+` ;
- **violet** : `theta2-`.

Le **gris** représente un état jamais visité ou sans action possible.
Les **deux configurations d'arrivée** sont entourées de cercles
blancs bordés de noir. Il n'y a pas de `S` sur le bras, puisque les
départs de l'entraînement sont aléatoires. Les graduations des angles
sont affichées tous les 60 degrés. Ces couleurs distinguent les actions ;
elles ne représentent ni une récompense ni la valeur de Q(s,a).

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
