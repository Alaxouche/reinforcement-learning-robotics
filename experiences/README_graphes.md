# Graphes des expériences Q-learning

Depuis la racine du dépôt (environ 7 minutes, dont l'essentiel pour les 10 seeds
du graphe 2) :

```powershell
py -m experiences.graphes
```

Le programme entraîne le labyrinthe 5x5 et les trois modes du bras avec les réglages
des scripts d'entraînement, puis écrit :

- `experiences/resultats/labyrinthes/`
- `experiences/resultats/bras_robotique/naive/`, `smart/` et `distance/`
- `experiences/resultats/comparaison_labyrinthe_bras.png`

Il affiche aussi les mesures de comparaison. Tous les graphes sauf le 2 utilisent
un seul agent (seed 42), identique à celui de `py -m experiences.entrainement` et de
`py -m experiences.entrainement_bras_robotique` : l'entraînement par blocs et le
contrôle du chemin ne changent aucun tirage aléatoire.

| Fichier | Contenu |
|---|---|
| `1_convergence.png` | Récompense, nombre de pas et epsilon par épisode |
| `2_epsilon_seeds.png` | 4 stratégies d'epsilon sur 10 seeds : répétabilité et épisode de convergence |
| `3_cartes_q_brutes.png` | Q(s, a) brute, une carte par action : rouge si négative, bleu si positive |
| `4_meilleure_action.png` | Meilleure action par état et chemin glouton depuis le départ de référence |
| `5_cartes_q.png` | Q(s, a) normalisée, une carte par action (Haut, Bas, Gauche, Droite pour le labyrinthe) |
| `5b_carte_valeurs.png` | V(s) normalisée sur les couples (theta1, theta2), bras seulement |
| `6_visites.png` | Nombre de passages par état (grille 36 x 36 pour le bras) |
| `7_carte_epsilon.png` | Epsilon moyen de chaque état, et epsilon en fonction de V(s) |
| `8_mises_a_jour.png` | Nombre de mises à jour de chaque action, une carte par action |
| `9_lien_epsilon_greedy.png` | Lien entre ces mises à jour et la stratégie epsilon-greedy |

Pour le bras, une ligne de grille est un angle theta1 et une colonne un angle
theta2 : chaque case est un couple d'angles, pas une position (x, y). Le bras n'a pas
de bord (theta revient à 0 après 350 degrés) : le chemin du graphe 4 se coupe à ce
passage au lieu de traverser la grille.

Le bras démarre d'un état aléatoire à chaque épisode. Le « S » des cartes est le
départ de référence (90, 0), qui ne sert qu'à l'évaluation.

## Lire les cartes

**Passages (6) et mises à jour (8).** L'échelle de couleur est logarithmique, car
l'arrivée et les abords du chemin reçoivent bien plus de passages que le reste. Le
gris signale un état jamais visité ou une action jamais mise à jour. Les deux cartes
se recoupent : chaque pas produit une mise à jour, et chaque épisode ajoute un passage
sur son état de départ.

**Carte d'epsilon (7).** Dans ce projet, epsilon ne dépend pas des Q-values : il
diminue à chaque épisode, de la même façon pour tous les états. La carte donne, pour
chaque état, la moyenne d'epsilon à ses passages. Elle sert d'horloge : un état clair a
été visité jusque tard, un état foncé surtout au début.

Le nuage de droite place chaque état selon V(s) et cet epsilon moyen. Le lien est
une tendance, pas une règle : la corrélation vaut -0.53 pour le labyrinthe, -0.54 pour
le bras naive, -0.16 pour le smart et -0.20 pour le distance. Ce n'est pas la Q-value
qui règle epsilon : les états de forte valeur sont souvent ceux que la politique
fréquente une fois epsilon descendu.

- **Labyrinthe :** deux groupes bien séparés. Les cases du chemin ont un epsilon moyen
  de 0.11 à 0.18, les autres de 0.42 à 0.78. La case ligne 3, colonne 4 montre qu'une
  forte valeur ne suffit pas : elle vaut 0.90 mais reste à 0.63, car elle est hors du
  chemin.
- **Bras :** pas de groupes, car chaque état sert aussi de départ. Les bandes claires
  sont les passages qui mènent aux deux arrivées. La plupart des points sont au-dessus
  de la ligne pointillée, car les premiers épisodes sont bien plus longs (49 pas en
  moyenne au début, 16 à 22 à la fin) et pèsent donc davantage dans la moyenne.

V(s) est calculée sur les actions essayées, car une action jamais essayée garde son
Q initial de 0.

## Lien avec epsilon-greedy

À chaque pas, l'agent choisit une action selon :

```text
P(a | s) = epsilon / |A(s)| + (1 - epsilon)   si a = argmax Q(s, .)
P(a | s) = epsilon / |A(s)|                   sinon
```

Le graphe 9 relie cette règle aux comptages du graphe 8 :

- **À gauche :** la part d'actions tirées au hasard suit la courbe d'epsilon, ce qui
  vérifie l'implémentation sur des mesures réelles.
- **Au milieu :** d'où viennent les mises à jour de chaque action. Dans le labyrinthe,
  Bas et Droite viennent de l'exploitation à 88 % et 86 %, Haut à seulement 3 %. Sur le
  bras, les quatre actions sont à 50-57 %, car aucune direction n'est meilleure
  partout.
- **À droite :** dans chaque état, l'action de plus forte Q-value finale reçoit le plus
  de mises à jour. C'est une boucle : une action bien notée est choisie plus souvent,
  donc mieux estimée, alors que les autres avancent au rythme de epsilon / |A(s)|.

## Comparaison avec le labyrinthe

Même algorithme, seed 42. Les lignes de réussite et de longueur portent sur la
politique gloutonne, sans exploration, depuis tous les états non terminaux.

| | Labyrinthe 5x5 | Bras naive | Bras smart | Bras distance |
|---|---:|---:|---:|---:|
| États utilisables | 22 | 1296 | 1296 | 1296 |
| Couples (s, a) autorisés | 56 | 5176 | 5176 | 5176 |
| Épisodes / pas maximum | 1000 / 30 | 5000 / 50 | 5000 / 50 | 5000 / 50 |
| Mises à jour par couple | 176.4 | 27.1 | 30.9 | 25.3 |
| États visités | 100 % | 100 % | 100 % | 100 % |
| Passages sur les 10 % d'états les plus visités | 21.8 % | 28.8 % | 22.4 % | 43.5 % |
| Actions tirées au hasard | 20.2 % | 47.3 % | 44.2 % | 47.0 % |
| Épisode où 90 % des 100 derniers réussissent | 129 | 2666 | 3410 | 1533 |
| **Réussite depuis tous les états** | **100 %** | **96.9 %** | **56.4 %** | **100 %** |
| Longueur moyenne des réussites / minimale | 4.5 / 4.1 | 12.8 / 12.5 | 10.3 / 9.8 | 14.8 / 12.8 |
| Depuis le départ de référence / minimum | 8 / 8 | 9 / 9 | 13 / 9 | 11 / 9 |

**Le problème n'est pas le même.** Le labyrinthe part toujours de la même case. Le bras
doit apprendre depuis n'importe quelle pose, et il y arrive en grande partie : tous ses
états sont visités et tous ses couples, à 0.7 % près, sont essayés. Les deux poses
d'arrivée servent d'ailleurs autant l'une que l'autre, avec environ 1800 arrivées
chacune en naive.

**La taille coûte du temps, pas de la couverture.** Le bras a 92 fois plus de couples
que le labyrinthe et reçoit 6 à 7 fois moins de mises à jour par couple. Le problème
est aussi plus long : jusqu'à 22 déplacements du départ à l'arrivée, contre 8 dans le
labyrinthe. Il atteint donc 90 % de réussite bien plus tard, après 1533 à 3410 épisodes
contre 129.

**L'exploration est réglée plus fort sur le bras.** Avec `epsilon_min=0.15` et
`decroissance=0.9995`, epsilon n'atteint son minimum qu'à l'épisode 3795 sur 5000, contre
300 sur 1000 pour le labyrinthe. Cela donne 44 à 47 % d'actions au hasard contre 20 %.
C'est ce qui permet de couvrir tout l'espace, au prix d'un entraînement moins exploité.

**Les mouvements sont répartis autrement.** Le labyrinthe concentre 90 % de ses mises à
jour sur Bas et Droite, car il faut toujours aller vers le bas à droite. Sur le bras,
les quatre actions reçoivent chacune environ un quart des mises à jour : la bonne direction
dépend de l'état de départ. Les différences apparaissent sur les cartes du graphe 8, pas
dans les totaux.

**Les trois récompenses du bras donnent trois comportements.**

- **Naive :** le meilleur compromis ici. Elle réussit 96.9 % des départs avec des
  chemins presque minimaux (3 % plus longs), mais elle apprend lentement.
- **Distance :** elle apprend le plus vite et réussit partout, mais avec des chemins 16 %
  plus longs, et seuls 55 % sont minimaux. C'est logique : la somme des récompenses
  vaut d(départ) - d(arrivée), quel que soit le chemin. Seul gamma favorise un peu les
  chemins courts. Elle ne tient pas compte de la longueur, contrairement au coût par pas
  du mode naive.
- **Smart :** elle n'est pas prête après 5000 épisodes. Elle échoue depuis 44 % des états,
  surtout loin de la cible, avec un résultat stable sur 5 seeds (53 à 60 %). Un test
  ponctuel avec plus d'épisodes donne 97 % à 20 000 épisodes et 100 % à 80 000 : c'est un
  manque d'entraînement, pas une récompense inutilisable. L'explication la plus probable
  est que ses Q-values partent de 0 alors qu'elles doivent descendre beaucoup (minimum
  -15 à 5000 épisodes, -31 à 80 000), ce qui prend du temps sur 5176 couples.

## Limites

- **L'épisode de convergence du graphe 2 ne se compare pas d'un mode à l'autre.** Il
  regarde seulement si le retour depuis le départ de référence change encore.
  - Mode distance : ce retour vaut toujours 1.41, la distance initiale, dès que la cible
    est atteinte. Le critère ne voit donc pas les changements de chemin, et ses 120 à
    460 épisodes ne prouvent pas que la politique est stable.
  - Mode smart : ce retour change au moindre écart de chemin. Ses 4845 à 4950 épisodes
    veulent seulement dire qu'il ne s'est pas stabilisé dans les 5000 épisodes.
- La seed 42 est plutôt favorable au mode naive depuis le départ de référence : 9
  mouvements, alors que le retour moyen sur 10 seeds (9.79) correspond à environ 11.
- Sauf le graphe 2 et le test sur 5 seeds du mode smart, les chiffres viennent d'une
  seule seed.
- Les récompenses des modes n'ont pas la même définition et ne se comparent pas
  numériquement. Le tableau utilise donc des mesures communes.
