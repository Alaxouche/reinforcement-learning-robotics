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
- `experiences/resultats/bras_robotique/<n>_actions/comparaison_reussite_apprentissage.png`
- `experiences/resultats/bras_robotique/<n>_actions/comparaison_longueur_episodes.png`
- `experiences/resultats/bras_robotique/comparaison_4_9_actions.png`

L'étude de epsilon (`py -m experiences.etude_epsilon`, ≈ 3 minutes) écrit ses
graphes dans `experiences/resultats/bras_robotique/epsilon/9_actions/`.

Les anciens fichiers `1_convergence.png`, `2_epsilon_seeds.png`,
`5_cartes_q.png`, `5b_carte_valeurs.png`, `7_carte_epsilon.png`,
`8_mises_a_jour.png` et `9_lien_epsilon_greedy.png` sont supprimés
lors de la prochaine génération : ils ne font plus partie de l'analyse.
Le PNG `comparaison_mouvements_finaux.png` (ancien boxplot) est également
supprimé automatiquement lors de la prochaine exécution, ainsi que les cartes
rangées dans les anciens dossiers `bras_robotique/<mode>/` et les comparaisons
placées directement dans `bras_robotique/` avant le passage à 4 / 9 actions.

## Graphiques conservés

| Fichier | Interprétation |
|---|---|
| `3_cartes_q_brutes.png` | Q(s,a) brute pour chaque action (2×2 cartes pour 4 actions, 3×3 pour 9), du bleu clair (faible) au bleu foncé (élevée) |
| `4_meilleure_action.png` | Meilleure action : flèches et chemin pour le labyrinthe, quatre couleurs pour le bras à 4 actions, deux cartes (variation de theta1, variation de theta2) pour le bras à 9 actions |
| `6_visites.png` | Couleur : passages par état. Dans chaque case : passages (haut) / mises à jour de Q (bas). L'écart vient du départ de chaque épisode (passage sans action préalable) et de l'arrivée (visitée, jamais mise à jour). Le 36×36 est une grande image à zoomer |
| `6b_passages_par_action.png` | Une carte par action (haut/bas/gauche/droite ; pour le bras theta1±, theta2±, plus les quatre combinaisons et l'action immobile à 9 actions, sur la grille des couples (theta1, theta2)) : nombre de fois où l'action a été jouée depuis chaque case. Même palette bleue et même échelle log pour comparer toutes les cartes |
| `7_ecart_optimal.png` | (labyrinthes) Si l'agent partait de chaque case et suivait sa politique apprise : bleu très clair = chemin optimal, bleu plus foncé = arrivée atteinte avec détour, orange pâle = la politique tourne en rond (case peu apprise). Le trait noir est le chemin appris depuis S. L'optimum est calculé exactement (Dijkstra sur les récompenses) |
| `comparaison_5x5_36x36.png` | Longueur des épisodes / plus court chemin, % de cases menant à l'arrivée, % de cases avec chemin optimal, et chemin depuis S, en fonction de la progression de l'entraînement (%) |
| `comparaison_longueur_episodes.png` | Bras : mouvements par épisode pendant l'entraînement (moyenne glissante sur 100), comparable entre récompenses |
| `comparaison_reussite_apprentissage.png` | Trois courbes de taux de réussite, tous les 500 épisodes, sur les mêmes 30 départs |
| `comparaison_4_9_actions.png` | Bras : réussite et longueur des épisodes, couleur = récompense, trait plein = 4 actions, tirets = 9 actions |
| `epsilon/.../couverture_couples.png` | Étude de epsilon : part des couples (s, a) essayés au moins une fois au fil de l'entraînement, une courbe par epsilon (bleu clair = faible, foncé = élevé), une colonne par initialisation de Q, une ligne par récompense |
| `epsilon/.../reussite.png` | Même disposition, réussite de la politique gloutonne sur les 30 départs de test |
| `epsilon/.../bilan_epsilon.png` | Mesures finales en fonction d'epsilon : couples essayés, réussite, longueur des 500 derniers épisodes ; bleu = Q initiale constante, orange = Q initiale aléatoire |

Les graphes 3, 4, 6 et 6b sont créés pour chaque environnement, le 7 pour les
labyrinthes. Les graphiques `comparaison_*` du bras comparent
**uniquement les trois récompenses du bras robotique**.

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

## Comparer les trois récompenses du bras

**Progression du taux de réussite :** les bras sont entraînés 5 000 épisodes.
Après chaque bloc de 500 épisodes, chacun est évalué **sans exploration**
sur les **mêmes 30 états initiaux non terminaux**, choisis une seule fois
avec une graine fixe (`42 + 10 000`). Chaque essai est limité à 50 mouvements.
La courbe mesure le pourcentage de départs ayant atteint la cible.
Les 10 points de chaque courbe correspondent aux épisodes 500, 1 000, ...
5 000 ; ce ne sont **pas des moyennes glissantes**.

Les évaluations sont déterministes : aucune exploration, aucun tirage
aléatoire, aucune mise à jour des Q-values ni des compteurs. Elles
n'interrompent pas réellement la continuité de l'apprentissage :
Q, epsilon et générateurs aléatoires sont conservés entre blocs.

## Limites d'interprétation

La Q-table n'est plus initialisée à zéro : chaque valeur Q(s,a) est
tirée uniformément dans [-0,1 ; 0,1]. Avec la graine `42` réappliquée
à chaque mode, les trois bras démarrent avec **la même table aléatoire**.
Les expériences doivent être relancées : les anciens PNG correspondent
à l'ancienne initialisation. Une action jamais essayée conserve toutefois
sa valeur initiale, et ce biais potentiel n'est pas complètement supprimé.

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
