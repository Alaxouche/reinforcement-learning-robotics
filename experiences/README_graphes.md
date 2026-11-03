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
- `experiences/resultats/bras_robotique/comparaison_reussite_apprentissage.png`
- `experiences/resultats/bras_robotique/comparaison_mouvements_finaux.png`

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
| `comparaison_reussite_apprentissage.png` | Trois courbes de taux de réussite, tous les 500 épisodes, sur les mêmes 30 départs |
| `comparaison_mouvements_finaux.png` | Boxplots des mouvements pour les essais réussis après 5 000 épisodes, avec le nombre d'échecs de chaque mode |

Les graphes 3, 4 et 6 sont créés pour chaque environnement. Les deux nouveaux
graphiques comparent **uniquement les trois récompenses du bras robotique**.
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

**Nombre de mouvements final :** après les 5 000 épisodes, le diagramme en
boîtes présente la distribution du nombre de mouvements **parmi les
essais réussis**. Le panneau inférieur indique explicitement combien
d'essais ont échoué (sur 30) : une politique qui ne réussit que les
départs faciles ne doit pas être considérée comme meilleure en raison
d'une faible médiane sur ses rares succès. Les losanges blancs indiquent
la moyenne et les traits dans les boîtes la médiane. Quand un mode n'a
aucune réussite, aucune boîte n'est dessinée.

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

La récompense `smart` utilise maintenant un **guidage par potentiel** :
`Phi(s) = -distance_cible(s)` hors arrivée et `Phi(arrivée) = 0`.
La récompense de base est `+10` à l'arrivée et `-0.02` sinon ; on ajoute
`0.95 * Phi(suivant) - Phi(courant)`. Ce terme transmet une information
sur la distance tout en conservant une incitation forte à terminer.
Le coefficient `0.95` est celui de `gamma` dans l'entraînement.
La théorie de l'invariance des politiques par guidage à potentiel s'applique
sous ses hypothèses ; nos épisodes sont néanmoins limités à 50 actions,
ce qui nécessite de vérifier expérimentalement les performances.

Les PNG générés avant ce changement sont obsolètes : relancer le script
pour produire les résultats correspondants.

Les récompenses des modes naive, smart et distance ont des échelles
différentes ; privilégier le taux de réussite et le nombre d'actions
pour les comparer.

Le programme utilise Matplotlib avec le moteur `Agg` : les images sont
**enregistrées sur disque**, sans fenêtre graphique.
