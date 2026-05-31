# Scripts de présentation — Hybrid ALNS

> Répartition basée sur les noms du fichier `LICENSE` : Mohamed El Amine Kherroubi, Rayan Boukakiou, Idris Yassine Ziadi, Idris Himeur, Adem Abdelhafidh Diar.
> Le script est organisé par sections, pas slide par slide, pour garder une présentation fluide et équitable.

---

## 1. Mohamed El Amine Kherroubi — Introduction et plan

Bonjour à tous. Aujourd’hui, nous présentons notre projet sur le problème de bin packing en une dimension, c’est-à-dire le rangement d’objets dans des boîtes.

Le principe est simple : on a des objets de tailles différentes, et on doit les placer dans le plus petit nombre possible de boîtes. Chaque boîte a une capacité maximale à ne pas dépasser.

Ce problème apparaît dans beaucoup de situations réelles : le chargement de véhicules, la logistique, la découpe industrielle, ou encore l’allocation de ressources dans le cloud.

Le bin packing appartient à la classe des problèmes fortement NP-difficiles. En pratique, cela veut dire qu’une méthode exacte peut trouver l’optimum, mais devient vite trop lente quand le nombre d’objets augmente. Les heuristiques classiques sont beaucoup plus rapides, mais elles utilisent souvent des règles fixes. Elles peuvent donc rester bloquées dans une bonne solution locale, sans réussir à l’améliorer.

Notre idée est d’ajouter de l’apprentissage automatique dans une métaheuristique, mais sans remplacer toute la méthode d’optimisation.

Dans notre approche, il y a deux décisions apprises :

- pendant la recherche, LinUCB fait une sélection adaptative de l’opérateur de destruction ;
- pendant la réparation, un modèle supervisé aide à choisir dans quelle boîte replacer chaque objet.

La présentation suit cette logique : définition du problème, contexte scientifique, solution proposée, tests expérimentaux, puis conclusion avec les limites et les pistes d’amélioration.

---

## 2. Rayan Boukakiou — Définition du problème et stratégie algorithmique

On commence par définir le problème plus précisément.

On a un ensemble d’objets. Chaque objet a une taille positive, et toutes les boîtes ont la même capacité. Une solution est faisable si chaque objet est placé une seule fois, et si la capacité de chaque boîte est respectée.

L’objectif est de minimiser le nombre total de boîtes utilisées.

Pour évaluer une solution, on utilise une borne inférieure simple. On additionne toutes les tailles des objets, on divise par la capacité d’une boîte, puis on arrondit au supérieur. Cette valeur donne le minimum théorique de boîtes nécessaires si l’espace était utilisé parfaitement.

Cette borne, appelée LB1, n’est pas toujours très serrée, mais elle donne un repère clair. Si notre solution atteint cette borne, alors elle est optimale par rapport à LB1.

Comme le problème est difficile à résoudre exactement à grande échelle, notre stratégie est heuristique. On construit d’abord une solution initiale avec Best-Fit Decreasing, puis on essaie de l’améliorer avec ALNS.

Best-Fit Decreasing trie les objets du plus grand au plus petit. Ensuite, pour chaque objet, il choisit la boîte faisable qui laisse le moins d’espace libre après insertion. Si aucune boîte ne convient, il ouvre une nouvelle boîte.

Cette étape donne un bon point de départ. Ensuite, ALNS peut chercher de meilleures combinaisons au lieu de partir d’une solution trop faible.

---

## 3. Idris Yassine Ziadi — Littérature et architecture globale

Avant de présenter notre méthode, on situe rapidement le projet dans la littérature.

Une première famille de travaux concerne la sélection adaptative d’opérateurs. Au lieu d’utiliser des probabilités fixes, la méthode apprend pendant l’exécution quels opérateurs sont les plus utiles. Dans notre projet, cette idée est implémentée avec LinUCB.

LinUCB appartient à la famille des méthodes de bandits contextuels. Le principe est le suivant : à chaque étape, on choisit une action, on observe une récompense, puis on améliore les choix suivants. Ici, les actions sont les opérateurs de destruction, et le contexte décrit l’état actuel de la recherche.

Une deuxième famille concerne la réparation ou la construction apprise. Le principe est d’entraîner un modèle à imiter une bonne heuristique. Dans notre cas, le modèle apprend à reproduire les choix de Best-Fit Decreasing pour replacer les objets.

Notre approche reste volontairement légère. On ne remplace pas ALNS par un modèle profond. On garde une métaheuristique connue, et on ajoute deux aides d’apprentissage à des endroits précis.

L’architecture globale est la suivante.

On part d’une instance avec des objets et une capacité. On génère une première solution avec BFD. Ensuite, la boucle ALNS commence.

À chaque itération, LinUCB reçoit un vecteur de contexte : température, stagnation, écart à la borne, taille de destruction, et avancement de la recherche. Avec ces informations, il choisit un opérateur de destruction.

Après la destruction, la solution partielle est réparée. Si le modèle GBT est activé, il attribue une note aux boîtes faisables pour chaque objet à réinsérer.

Enfin, on applique le critère d’acceptation du recuit simulé. La meilleure solution trouvée pendant toute la recherche est conservée et retournée à la fin.

---

## 4. Idris Himeur — ALNS, destruction et acceptation

Je vais maintenant expliquer la partie métaheuristique.

Une recherche locale classique fait de petits changements, par exemple déplacer un seul objet. C’est rapide, mais cela peut mener à un optimum local : aucune petite modification n’améliore la solution, même si une meilleure solution existe ailleurs.

ALNS répond à ce problème avec deux étapes : destruction et réparation.

Pendant la destruction, on retire une partie des objets déjà placés. Pendant la réparation, on les réinsère pour obtenir une nouvelle solution complète. Comme plusieurs objets sont modifiés en même temps, la méthode explore un voisinage beaucoup plus large qu’une recherche locale simple.

Dans notre solveur, il y a trois opérateurs de destruction.

Le premier est aléatoire : il retire des objets au hasard. Il sert surtout à explorer différentes zones de recherche.

Le deuxième cible les boîtes les moins remplies. L’idée est de modifier les parties faibles de la solution, car elles sont souvent responsables d’un mauvais remplissage.

Le troisième retire des objets de tailles proches. Cela permet de reconstruire des groupes d’objets similaires, qui peuvent parfois être mieux combinés autrement.

Le nombre d’objets retirés varie à chaque itération. En général, on retire entre 5 % et 25 % des objets. Si la recherche stagne, ce nombre peut augmenter pour forcer une diversification plus forte.

Pour accepter ou refuser une nouvelle solution, on utilise le recuit simulé. Une solution meilleure est acceptée directement. Une solution moins bonne peut aussi être acceptée avec une certaine probabilité, surtout au début, quand la température est élevée.

La température diminue progressivement. Si la recherche stagne trop longtemps, on peut la réchauffer légèrement. Et en cas de stagnation prolongée, on repart de la meilleure solution trouvée jusque-là.

Le but est de garder un équilibre entre exploitation des bonnes solutions et exploration de nouvelles possibilités.

---

## 5. Adem Abdelhafidh Diar — Composants d’apprentissage automatique

Je présente maintenant les deux composants d’apprentissage automatique.

Le premier composant choisit l’opérateur de destruction. Il utilise LinUCB, une méthode de sélection adaptative basée sur le contexte. Son rôle est de gérer le compromis entre exploration et exploitation : tester différents opérateurs, mais aussi réutiliser ceux qui ont déjà donné de bons résultats.

Au début, on utilise une phase de démarrage avec l’échantillonnage de Thompson bêta-bernoulli pendant les 300 premiers appels. Cette phase explore les trois opérateurs sans dépendre trop vite du modèle contextuel.

Ensuite, LinUCB prend le relais. Pour chaque opérateur, il calcule une note à partir du contexte courant. Cette note combine la performance estimée et une marge d’exploration. L’opérateur avec la meilleure note est choisi.

La récompense dépend de ce qui se passe après l’itération. Si on réduit le nombre de boîtes, la récompense est élevée. Si la solution est acceptée sans réduire le nombre de boîtes, la récompense est faible. Si elle est refusée, la récompense est nulle.

Le deuxième composant est le modèle de réparation appris. Il utilise un classifieur Gradient Boosting.

Après une destruction, chaque objet retiré doit être replacé. Pour chaque objet, on regarde toutes les boîtes dans lesquelles il peut être placé. Le modèle donne une note à chaque paire objet-boîte, puis on choisit la boîte avec la meilleure note.

Le modèle utilise 11 caractéristiques, par exemple la taille de l’objet, la charge actuelle de la boîte, l’espace restant, l’espace après insertion, et des informations sur les objets déjà présents dans la boîte.

L’entraînement se fait par imitation de Best-Fit Decreasing. Quand BFD choisit une boîte, cette paire reçoit l’étiquette positive. Les autres boîtes faisables reçoivent l’étiquette négative.

Pour réduire le décalage entre entraînement et utilisation réelle, on entraîne le modèle avec deux types de traces : des traces BFD complètes, et des traces obtenues après destruction partielle d’une solution.

---

## 6. Mohamed El Amine Kherroubi — Protocole expérimental et jeux de données

Pour évaluer la méthode, on utilise le même protocole pour tous les tests.

La graine aléatoire est fixée à 42. ALNS utilise 300 itérations. La température initiale vaut 1 sur ln 2, et le coefficient de refroidissement est 0,9995.

Pour LinUCB, le paramètre d’exploration alpha vaut 0,3, avec 300 appels de phase de démarrage. Pour la réparation apprise, on utilise un seul modèle pré-entraîné sur tous les jeux de données.

La métrique principale est l’écart à la borne : nombre de boîtes utilisées moins LB1. Un écart de zéro veut dire que la solution atteint la borne inférieure utilisée.

On teste sur cinq familles de jeux de données de référence.

Scholl-2 contient des tailles plutôt uniformes. Falkenauer-T a une structure en triplets, ce qui rend LB1 moins fiable. Falkenauer-U contient aussi des tailles uniformes, mais avec une autre capacité. Wäscher est lié aux problèmes de découpe. Enfin, Hard28 contient des instances volontairement difficiles.

Cette diversité permet de vérifier si la méthode reste stable sur plusieurs structures d’instances, et pas seulement sur un cas particulier.

---

## 7. Rayan Boukakiou — Résultats : ablation et généralisation

Le premier test est une étude d’ablation. Le but est de mesurer séparément l’effet de LinUCB et celui du modèle de réparation GBT.

Sur Scholl-2, avec cinq instances de 50 objets, la version sans apprentissage obtient un écart moyen de 0,20.

Quand on active seulement LinUCB, l’écart moyen passe à 0,00. C’est le meilleur résultat dans cette étude, avec un temps de calcul encore faible.

Quand on active seulement le modèle GBT, l’écart reste à 0,20, mais le temps augmente. Avec les deux composants ensemble, l’écart reste aussi à 0,20 sur ce petit échantillon.

La conclusion est donc que, dans ces tests, le choix adaptatif des opérateurs est le composant le plus utile. Le modèle GBT est bien intégré, mais son coût n’est pas encore compensé par un gain de qualité sur les petites instances.

Le deuxième test regarde la généralisation sur plusieurs familles.

La méthode combinée reste à moins d’une boîte de LB1 sur quatre familles sur cinq. Sur Falkenauer-T, l’écart moyen est de 1,00, mais cette famille a une structure en triplets qui affaiblit LB1. Cet écart n’indique donc pas forcément une mauvaise recherche.

Globalement, les résultats montrent que la méthode est assez robuste, même sans réglage spécifique pour chaque famille.

---

## 8. Idris Yassine Ziadi — Résultats comparatifs

Le troisième test compare notre méthode avec plusieurs approches classiques.

On compare avec FFD et BFD, le recuit simulé, la recherche tabou, un algorithme génétique, l’optimisation par colonies de fourmis, et notre ALNS hybride avec apprentissage.

Sur Scholl-2 avec cinq instances de 50 objets, les méthodes constructives sont très rapides, mais elles gardent un écart moyen de 1,80.

Le recuit simulé et la recherche tabou ont aussi un écart de 1,80 dans ce test. L’algorithme génétique améliore le résultat avec un écart de 1,00, mais il prend plus de temps.

Notre ALNS hybride obtient un écart moyen de 0,20 en environ 0,36 seconde.

L’optimisation par colonies de fourmis obtient un écart de 0,00, donc une meilleure qualité sur ce test, mais avec un temps moyen d’environ 3,03 secondes. Elle est donc beaucoup plus lente que notre méthode.

Notre approche n’est pas toujours la meilleure en qualité pure, mais elle donne un très bon compromis entre qualité et temps de calcul.

Il faut rester prudent, car les méthodes stochastiques sont évaluées sur une seule exécution. Avec plusieurs graines aléatoires, les classements pourraient changer.

---

## 9. Idris Himeur — Synthèse, limites et conclusion

Pour résumer, on retient plusieurs points.

Premièrement, le choix adaptatif des opérateurs est le composant d’apprentissage le plus important dans nos tests. Il améliore clairement l’écart dans l’étude d’ablation.

Deuxièmement, le modèle de réparation GBT fonctionne, mais il n’apporte pas encore de gain visible sur les petites instances. Pour l’instant, il ajoute surtout du temps de calcul.

Troisièmement, la méthode combinée se généralise correctement sur plusieurs familles de jeux de données de référence.

Quatrièmement, les deux composants d’apprentissage n’ont pas le même impact. C’est une information importante pour améliorer la suite du projet.

Enfin, par rapport aux métaheuristiques classiques testées, notre méthode offre un bon compromis entre qualité et vitesse.

Les limites sont aussi importantes. Les tests utilisent seulement de petits échantillons de cinq instances. Plusieurs comparaisons sont faites avec une seule exécution. Et la borne LB1 est simple ; une borne plus forte, comme Martello-Toth L2, permettrait une évaluation plus précise.

En conclusion, notre contribution est une ALNS hybride pour le bin packing 1D, avec deux composants activables séparément : LinUCB pour choisir les opérateurs de destruction, et un modèle GBT supervisé pour guider la réparation.

Les prochaines étapes seraient de tester plus d’instances, de faire des moyennes sur plusieurs graines, d’utiliser une meilleure borne inférieure, et d’améliorer le modèle de réparation.

---

## 10. Adem Abdelhafidh Diar — Mot de fin

Pour finir, notre objectif n’était pas de remplacer les heuristiques classiques par un modèle opaque.

Nous avons plutôt gardé une méthode d’optimisation connue, ALNS, et ajouté de l’apprentissage seulement aux décisions répétées où cela peut aider.

Les résultats montrent que cette idée est prometteuse, surtout pour le choix adaptatif des opérateurs.

Merci pour votre attention. Nous sommes prêts à répondre à vos questions.
