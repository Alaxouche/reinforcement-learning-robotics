# Graphiques des expériences Q-learning

Depuis la racine du dépôt :

```powershell
py -m pip install -r requirements.txt
py -m experiences.graphes
```

Le programme entraîne les **labyrinthes 5×5 et 36×36** et les trois modes du bras
(**naive, smart, distance**), à **4 puis à 9 actions**, avec la graine `42`.
Il enregistre des PNG dans :

- `experiences/resultats/labyrinthes/5x5/` et `experiences/resultats/labyrinthes/36x36/`
- `experiences/resultats/labyrinthes/comparaison_5x5_36x36.png`
- `experiences/resultats/bras_robotique/<n>_actions/naive/`, `.../smart/`, `.../distance/`
  (`<n>` = 4 ou 9)

Les anciens fichiers `1_convergence.png`, `2_epsilon_seeds.png`,
`5_cartes_q.png`, `5b_carte_valeurs.png`, `7_carte_epsilon.png`,
`8_mises_a_jour.png` et `9_lien_epsilon_greedy.png` sont supprimés
lors de la prochaine génération : ils ne font plus partie de l'analyse.
Le PNG `comparaison_mouvements_finaux.png` (ancien boxplot) est également
supprimé automatiquement lors de la prochaine exécution, ainsi que les cartes
rangées dans les anciens dossiers `bras_robotique/<mode>/`.
Les anciens graphiques comparant réussite et mouvements des trois récompenses
et leur comparaison 4/9 actions ne sont plus générés ; les PNG locaux
correspondants sont effacés automatiquement.

## Graphiques conservés

| Fichier | Interprétation |
|---|---|
| `3_cartes_q_brutes.png` | Q(s,a) brute pour chaque action (2×2 cartes pour 4 actions, 3×3 pour 9), du bleu clair (faible) au bleu foncé (élevée) |
| `4_meilleure_action.png` | Meilleure action : flèches et chemin pour le labyrinthe, quatre couleurs pour le bras à 4 actions, deux cartes (variation de theta1, variation de theta2) pour le bras à 9 actions |
| `6_visites.png` | Couleur : passages par état. Dans chaque case : passages (haut) / mises à jour de Q (bas). L'écart vient du départ de chaque épisode (passage sans action préalable) et de l'arrivée (visitée, jamais mise à jour). Le 36×36 est une grande image à zoomer |
| `6b_passages_par_action.png` | Une carte par action (haut/bas/gauche/droite ; pour le bras theta1±, theta2±, plus les quatre combinaisons et l'action immobile à 9 actions, sur la grille des couples (theta1, theta2)) : nombre de fois où l'action a été jouée depuis chaque case. Même palette bleue et même échelle log pour comparer toutes les cartes |
| `7_ecart_optimal.png` | (labyrinthes) Si l'agent partait de chaque case et suivait sa politique apprise : bleu très clair = chemin optimal, bleu plus foncé = arrivée atteinte avec détour, orange pâle = la politique tourne en rond (case peu apprise). Le trait noir est le chemin appris depuis S. L'optimum est calculé exactement (Dijkstra sur les récompenses) |
| `comparaison_5x5_36x36.png` | Longueur des épisodes / plus court chemin, % de cases menant à l'arrivée, % de cases avec chemin optimal, et chemin depuis S, en fonction de la progression de l'entraînement (%) |

Les graphes 3, 4, 6 et 6b sont créés pour chaque environnement, le 7 pour
les labyrinthes. Les courbes de réussite et de mouvements par épisode du bras
ont été supprimées.

**Pourquoi des cases orange dans `7_ecart_optimal.png` du 36×36 ?** L'agent
commence toujours en S. Une fois epsilon faible, il ne parcourt presque plus
que son trajet : les cases éloignées gardent des Q-values peu apprises et, en
partant de là, la politique gloutonne peut boucler. Ce n'est pas une erreur
de l'algorithme mais une limite de l'exploration depuis un départ unique ;
des départs aléatoires corrigeraient ces zones.
Le graphique de comparaison labyrinthe/bras a été retiré. Les indicateurs
de synthèse restent affichés dans le terminal.

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

Pour le **bras à 9 actions**, neuf couleurs catégorielles ne seraient pas
distinguables. La meilleure action est donc décomposée en deux cartes côte
à côte : la variation choisie pour `theta1` et celle choisie pour `theta2`.
**Bleu** = l'angle augmente, **rouge** = il diminue, **gris très clair** =
l'articulation ne bouge pas. Une case bleue ou rouge sur les deux cartes
correspond à un mouvement simultané ; gris très clair sur les deux, à l'action
immobile. Le titre indique la part des états visités pour chacun de ces cas.

**Graphe 6 :** les couleurs utilisent une **échelle log**
pour rendre visibles les écarts de comptage entre les états peu et très
fréquentés. Le gris indique un état jamais visité. Les visites
comptent aussi l'état initial de chaque épisode.

## Étude epsilon et départ du bras

Lancer `py -m experiences.etude_epsilon_depart` pour comparer les positions
initiales **fixes (90°, 0°)** et **aléatoires**, à Q(s,a)=0 partout, suivant
trois vitesses de décroissance d'epsilon (0.99, 0.997, 0.9995).
Les trois figures (`epsilon.png`, `exploration_effective.png`
et `couverture.png`) sont rangées dans
`experiences/resultats/bras_robotique/epsilon_depart/<n>_actions/<mode>/`.
Par défaut : récompense distance, 9 actions, 5 000 épisodes et trois graines.
Les évaluations gloutonnes utilisent les mêmes départs de test pour les
deux stratégies ; la comparaison ne confond donc pas explorations et
performances de la politique apprise.

## Limites d'interprétation

La Q-table est désormais initialisée à zéro pour **tous** les agents.
Cela change les conditions des expériences : les graphiques produits
avant cette modification sont obsolètes et doivent être régénérés.
Les départs aléatoires ne concernent que la **position du robot**, jamais
l'initialisation de Q.

Les comparaisons reposent ici sur une seule graine (`42`) ; elles
permettent de décrire une expérience, mais ne prouvent pas la convergence
mathématique des Q-values ni la supériorité générale d'une récompense.

La récompense `smart` utilise à nouveau la formule simple
`-distance_apres + (10 si termine sinon 0)` : elle favorise une faible
distance à la cible après chaque action, avec un bonus de réussite de `+10`.
Elle ne mesure pas la variation de distance ; c'est le rôle distinct
du mode `distance`.

Ce retour à une formule simple n'implique pas que ses résultats soient
meilleurs : les graphiques doivent être régénérés pour l'évaluer.

Les PNG générés avant ce changement sont obsolètes : relancer le script
pour produire les résultats correspondants.

Les récompenses des modes naive, smart et distance ont des échelles
différentes ; privilégier le taux de réussite et le nombre d'actions
pour les comparer.

Le programme utilise Matplotlib avec le moteur `Agg` : les images sont
**enregistrées sur disque**, sans fenêtre graphique.
