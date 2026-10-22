# Graphiques des expériences Q-learning

Tous les graphiques sont sauvegardés automatiquement au format PNG.
Les labyrinthes sont enregistrés dans `experiences/resultats/labyrinthes/` ;
les deux modes du bras robotique ont des dossiers séparés sous
`experiences/resultats/bras_robotique/`.

## Entraînement normal

Depuis la racine du dépôt :

```powershell
py -m experiences.entrainement
py -m experiences.entrainement_bras_robotique
```

Chaque entraînement crée les **six graphiques génériques** de
`experiences/graphiques.py`, puis les **sept nouveaux graphiques**
de `experiences/graphes_analyse.py` :

- `10_convergence_triptyque.png` : récompense cumulée, nombre d'actions et epsilon, sur un même graphique (3 panneaux). Récompense et actions ont une moyenne glissante.
- `11_surfaces_q_actions.png` : Q-values en 3D pour les quatre actions (4 vues). Pour le labyrinthe : abscisses des cases ; pour le bras : angles des articulations, en degrés.
- `12_meilleure_action.png` : meilleure action Q par état et chemin glouton en orange. Une politique gloutonne peut échouer ou boucler : c'est indiqué.
- `13_q_*.png`, `14_q_*.png`, `15_q_*.png`, `16_q_*.png` : **quatre cartes bleues indépendantes, une par action**, et PAS une carte globale `V(s) = max Q(s,a)`. La barre de couleur est la même sur toutes les cartes, afin de comparer honnêtement les actions. Les états terminaux/actions impossibles sont gris.

Pour le labyrinthe, les quatre cartes correspondent à `Haut`, `Bas`,
`Gauche`, `Droite`. Pour le bras : `theta1 +`, `theta1 -`, `theta2 +`,
`theta2 -`. Les cartes du robot se lisent dans l'espace des **angles**
(theta1, theta2), pas dans le plan physique (x,y). Le tracé de l'effecteur
reste dans `07_trajectoire_bras.png`.

## Comparaison d'epsilon sur plusieurs seeds (expérience séparée)

Le fichier fourni qui comparait 4 explorations × 10 seeds utilisait
des anciennes classes et pourrait nécessiter des dizaines d'entraînements.
Cette analyse est donc **optionnelle**, et produit `17_epsilon_seeds.png` :

```powershell
py -m experiences.comparer_exploration --environnement labyrinthe
py -m experiences.comparer_exploration --environnement bras --mode smart
py -m experiences.comparer_exploration --environnement bras --mode naive
```

Par défaut, quatre stratégies d'epsilon et **trois seeds** sont comparées.
Pour aller plus vite lors d'un essai :

```powershell
py -m experiences.comparer_exploration --environnement labyrinthe --seeds 2 --episodes 100
```

- À gauche : récompenses à l'entraînement, moyenne ± écart-type entre seeds.
- À droite : résultats finaux **sans exploration** pour chaque seed.

On compare uniquement des récompenses produites par **la même définition
de récompense** ; il ne faut pas comparer numériquement les scores du
mode `naive` et du mode `smart`.

La « convergence » des fichiers d'origine était déterminée par la
constance d'un score de parcours d'un bloc à l'autre. Une récompense
stable, même faible, ne prouve pas que la Q-table a convergé ni que la
politique est optimale : la boîte de droite montre donc les scores
finaux réels plutôt qu'un faux épisode de convergence.

Les labyrinthes 3×3 et 9×9 mentionnés dans les fichiers d'origine ne
font pas partie du dépôt actuel. Les fonctions d'analyse restent
compatibles avec de futurs labyrinthes carrés fondés sur la classe
`Labyrinthe`, sans changement des tracés.
