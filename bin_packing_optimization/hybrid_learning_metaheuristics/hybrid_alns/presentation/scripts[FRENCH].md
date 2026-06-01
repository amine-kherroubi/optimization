# Scripts de présentation — Hybrid ALNS

> Répartition basée sur les noms du fichier `LICENSE` : Mohamed El Amine Kherroubi, Rayan Boukakiou, Idris Yassine Ziadi, Idris Himeur, Adem Abdelhafidh Diar.
>
> **Indicateur d'utilisation :** chaque bloc commence par `Slide actuel : X` pour indiquer précisément le slide à afficher pendant que le texte est prononcé. Les slides de séparation sont indiqués explicitement et servent surtout de transition rapide.

---

## 1. Mohamed El Amine Kherroubi — Introduction et plan

> **Slides couverts : 1 à 3.**

**Slide actuel : 1 — Titre**

Bonjour à tous. Aujourd’hui, nous présentons notre projet sur le problème de bin packing en une dimension, c’est-à-dire le rangement d’objets dans des boîtes de capacité fixe.

Notre solution s’appelle Hybrid ALNS, parce qu’elle combine une métaheuristique de recherche par grands voisinages, une sélection adaptative des opérateurs, et une réparation guidée par apprentissage supervisé.

**Slide actuel : 2 — Introduction**

Le principe du bin packing est simple : on a des objets de tailles différentes, et on doit les placer dans le plus petit nombre possible de boîtes, sans dépasser la capacité de chaque boîte.

Ce problème apparaît dans beaucoup de situations réelles : la découpe industrielle, le chargement de véhicules, la logistique de conteneurs, ou encore l’allocation de ressources dans le cloud.

Il est fortement NP-difficile. En pratique, cela veut dire qu’une méthode exacte peut trouver l’optimum, mais devient vite trop lente quand le nombre d’objets augmente. Les heuristiques classiques sont beaucoup plus rapides, mais elles utilisent souvent des règles fixes. Elles peuvent donc rester bloquées dans une bonne solution locale, sans réussir à l’améliorer.

Notre idée est d’ajouter de l’apprentissage automatique dans une métaheuristique, mais sans remplacer toute la méthode d’optimisation.

Dans notre approche, il y a deux décisions apprises :

- en ligne, LinUCB apprend quel opérateur de destruction utiliser selon l’état courant de la recherche ;
- hors ligne, un modèle supervisé apprend dans quelle boîte replacer chaque objet déplacé, en imitant Best-Fit Decreasing.

Ces deux composants peuvent être activés ou désactivés séparément, ce qui permet de faire une étude d’ablation propre.

**Slide actuel : 3 — Plan**

La présentation suit le plan affiché : définition du problème, revue de littérature, solution proposée, tests expérimentaux, puis synthèse avec les limites et les pistes d’amélioration.

---

## 2. Rayan Boukakiou — Définition du problème et stratégie algorithmique

> **Slides couverts : 4 à 6.**

**Slide actuel : 4 — Définition formelle**

On commence par définir le problème plus précisément.

On a un ensemble d’objets. Chaque objet a une taille entière positive, et toutes les boîtes ont la même capacité entière. Une solution est faisable si chaque objet est placé une seule fois, et si la capacité de chaque boîte est respectée.

L’objectif est de minimiser le nombre total de boîtes utilisées.

**Slide actuel : 5 — Borne inférieure LB1**

Pour évaluer une solution, on utilise une borne inférieure simple. On additionne toutes les tailles des objets, on divise par la capacité d’une boîte, puis on arrondit au supérieur.

Cette valeur, appelée LB1, donne le minimum théorique de boîtes nécessaires si tout le volume était utilisé parfaitement. Elle n’est pas toujours très serrée, mais elle donne un repère clair. Si une solution utilise exactement LB1 boîtes, alors elle atteint une borne inférieure et elle est donc optimale pour cette instance.

Il existe des bornes plus fortes, comme Martello-Toth L2, mais dans ce projet LB1 suffit comme indicateur de progression.

**Slide actuel : 6 — Complexité et stratégie**

Comme le bin packing 1D est fortement NP-difficile, les méthodes exactes ont des garanties fortes mais ne passent pas bien à l’échelle. Les méthodes d’approximation comme FFD ou BFD sont rapides, mais restent basées sur des règles fixes. Les métaheuristiques, comme ALNS, n’ont pas de garantie d’optimalité, mais elles sont adaptées aux instances plus grandes.

Notre stratégie est donc la suivante : on construit d’abord une solution initiale déterministe avec Best-Fit Decreasing, puis on utilise ALNS pour essayer d’échapper aux optima locaux et se rapprocher de la borne inférieure.

---

## 3. Idris Yassine Ziadi — Littérature, architecture globale et initialisation

> **Slides couverts : 7 à 10.**

**Slide actuel : 7 — Revue de littérature**

Avant de présenter notre méthode, on situe rapidement le projet dans la littérature.

Une première famille de travaux concerne la sélection adaptative d’opérateurs. Au lieu d’utiliser des probabilités fixes, la méthode apprend pendant l’exécution quels opérateurs sont les plus utiles. Dans notre projet, cette idée est implémentée avec LinUCB, un bandit contextuel.

Le principe est le suivant : à chaque étape, on choisit une action, on observe une récompense, puis on améliore les choix suivants. Ici, les actions sont les opérateurs de destruction, et le contexte décrit l’état actuel de la recherche.

Une deuxième famille concerne la réparation ou la construction apprise. Le principe est d’entraîner un modèle à imiter une bonne heuristique. Dans notre cas, le modèle apprend à reproduire les choix de Best-Fit Decreasing pour replacer les objets dans une solution partielle.

Une troisième famille utilise l’apprentissage dans des méthodes de type Large Neighborhood Search. Notre approche reste volontairement plus légère et plus interprétable : on ne remplace pas ALNS par un modèle profond, on ajoute deux aides d’apprentissage à des décisions précises.

**Slide actuel : 8 — Séparation : solution proposée**

On passe maintenant à la solution proposée.

**Slide actuel : 9 — Architecture globale**

L’architecture globale est la suivante.

On part d’une instance avec des objets et une capacité. On génère d’abord une solution avec BFD. Ensuite, la boucle ALNS commence.

À chaque itération, le sélecteur choisit un opérateur de destruction. Si LinUCB est activé, ce choix dépend du contexte courant : température normalisée, stagnation, rapport entre LB1 et le coût courant, taille de destruction, et avancement global de la recherche.

Après la destruction, la solution partielle est réparée. Si le modèle GBT est activé, il attribue une note à chaque boîte faisable pour chaque objet à réinsérer.

Enfin, on applique le critère d’acceptation du recuit simulé. La meilleure solution trouvée pendant toute la recherche est conservée, et la performance finale est mesurée par l’écart entre le nombre de boîtes et LB1.

**Slide actuel : 10 — Initialisation BFD**

Best-Fit Decreasing sert de point de départ déterministe. Il trie les objets du plus grand au plus petit. Ensuite, pour chaque objet, il choisit la boîte faisable qui laisse le moins d’espace libre après insertion. Si aucune boîte ne convient, il ouvre une nouvelle boîte.

Cette étape donne un point de départ assez compact. Ensuite, ALNS peut chercher de meilleures combinaisons au lieu de partir d’une solution initiale trop faible.

---

## 4. Idris Himeur — ALNS, destruction et acceptation

> **Slides couverts : 11 à 18.**

**Slide actuel : 11 — Séparation : cadre métaheuristique**

Je vais maintenant expliquer la partie métaheuristique : Adaptive Large-Neighborhood Search.

**Slide actuel : 12 — Limite de la recherche locale**

Une recherche locale classique fait de petits changements, par exemple déplacer un seul objet. C’est rapide, mais cela peut mener à un optimum local : aucune petite modification n’améliore la solution, même si une meilleure solution existe ailleurs.

Dans le bin packing, ces optima locaux sont fréquents, parce que la qualité dépend beaucoup des groupements d’objets dans les boîtes.

**Slide actuel : 13 — Itération LNS**

ALNS répond à ce problème avec deux étapes : destruction et réparation.

Pendant la destruction, on retire une partie des objets déjà placés. Pendant la réparation, on les réinsère pour obtenir une nouvelle solution complète. Comme plusieurs objets sont modifiés en même temps, la méthode explore un voisinage beaucoup plus large qu’une recherche locale simple.

ALNS ajoute une idée supplémentaire : choisir adaptativement l’opérateur de destruction dans un portefeuille d’opérateurs.

**Slide actuel : 14 — Opérateurs de destruction**

Dans notre solveur, il y a trois opérateurs de destruction.

Le premier est aléatoire : il retire des objets uniformément au hasard parmi les objets placés. Il sert surtout à explorer différentes zones de recherche.

Le deuxième cible les boîtes les moins remplies. L’idée est de modifier les parties faibles de la solution, car elles sont souvent responsables d’un mauvais remplissage.

Le troisième retire des objets de tailles proches d’un objet choisi comme graine. Cela permet de reconstruire des groupes d’objets similaires, qui peuvent parfois être mieux combinés autrement.

Les trois opérateurs garantissent qu’au moins un objet reste placé.

**Slide actuel : 15 — Rayon de destruction**

Le nombre d’objets retirés varie à chaque itération. En général, on retire entre 5 % et 25 % des objets.

Si la recherche stagne, la borne supérieure du nombre d’objets retirés augmente progressivement. Cela force une diversification plus forte quand les petites modifications ne suffisent plus.

**Slide actuel : 16 — Séparation : mécanisme d’acceptation**

Après avoir créé une nouvelle solution, il faut décider si on l’accepte ou non. Pour cela, on utilise un mécanisme inspiré du recuit simulé.

**Slide actuel : 17 — Acceptation et refroidissement**

Une solution meilleure ou égale est acceptée directement. Une solution moins bonne peut aussi être acceptée avec une probabilité qui dépend de la température : au début, la température est plus élevée, donc on explore davantage ; ensuite, la température diminue progressivement.

Le refroidissement est géométrique. Si la recherche stagne, on peut aussi réchauffer légèrement la température pour éviter qu’elle devienne trop faible trop vite.

**Slide actuel : 18 — Redémarrage de diversification**

En cas de stagnation prolongée, on repart de la meilleure solution trouvée jusque-là. La température est réchauffée, la fenêtre de patience diminue, et le compteur de stagnation est remis à zéro.

Ce mécanisme ne termine pas l’algorithme. Le seul critère d’arrêt dur est le nombre maximal d’itérations. Le but est de garder un équilibre entre exploitation des bonnes solutions et exploration de nouvelles possibilités.

---

## 5. Adem Abdelhafidh Diar — Composants d’apprentissage automatique

> **Slides couverts : 19 à 27.**

**Slide actuel : 19 — Séparation : composants ML**

Je présente maintenant les deux composants d’apprentissage automatique : la sélection d’opérateurs et la réparation apprise.

**Slide actuel : 20 — Deux composants indépendants**

Le premier composant choisit l’opérateur de destruction à chaque itération. Le deuxième composant choisit la boîte dans laquelle replacer chaque objet déplacé.

Ces deux composants sont indépendants dans l’implémentation. On peut donc tester seulement LinUCB, seulement le modèle de réparation, les deux ensemble, ou aucun des deux.

**Slide actuel : 21 — Phase 1 : démarrage Thompson Sampling**

Au début, on utilise une phase de démarrage avec l’échantillonnage de Thompson bêta-bernoulli pendant les 300 premiers appels.

Les trois opérateurs commencent avec une loi Beta uniforme. À chaque appel, on échantillonne une valeur pour chaque opérateur, on choisit le meilleur échantillon, puis on met à jour seulement l’opérateur choisi à partir de la récompense observée.

Cette phase explore les trois opérateurs sans dépendre trop vite du vecteur de contexte. Elle réduit donc le problème de démarrage à froid de LinUCB.

**Slide actuel : 22 — Phase 2 : LinUCB**

Après les 300 premiers appels, LinUCB prend le relais. Pour chaque opérateur, il calcule une note à partir du contexte courant. Cette note combine la performance estimée et une marge d’exploration contrôlée par alpha, ici égal à 0,3.

L’opérateur avec la meilleure note est choisi. Après l’itération, les paramètres de l’opérateur choisi sont mis à jour avec la récompense obtenue.

**Slide actuel : 23 — Vecteur de contexte**

Le contexte contient cinq caractéristiques normalisées entre 0 et 1 : la température relative, la progression de la stagnation, le rapport LB1 sur le coût courant, le rayon de destruction relatif, et la progression globale dans le budget d’itérations.

Ces variables décrivent l’état de la recherche, ce qui permet à LinUCB de choisir des opérateurs différents selon la situation.

**Slide actuel : 24 — Récompense**

La récompense dépend de ce qui se passe après l’itération. Si on réduit le nombre de boîtes, la récompense est proportionnelle au gain, normalisé par l’écart courant à LB1 et borné à 1. Si la solution est acceptée sans réduire le nombre de boîtes, la récompense vaut 0,2. Si elle est refusée, la récompense vaut 0.

Cette normalisation donne plus de valeur aux gains obtenus quand la solution est déjà proche de la borne inférieure.

**Slide actuel : 25 — Réparation apprise**

Le deuxième composant est le modèle de réparation appris. Après une destruction, chaque objet retiré doit être replacé.

Pour chaque objet, on regarde toutes les boîtes dans lesquelles il peut être placé. S’il n’y a aucune boîte faisable, on ouvre une nouvelle boîte. Sinon, le modèle donne une note à chaque paire objet-boîte, puis on choisit la boîte avec la meilleure note.

Les objets sont réinsérés du plus grand au plus petit, comme dans Best-Fit Decreasing.

**Slide actuel : 26 — Représentation des caractéristiques**

Le modèle utilise 11 caractéristiques normalisées, par exemple la taille de l’objet, son rang, la fraction d’objets restant à réinsérer, la charge actuelle de la boîte, l’espace restant, l’espace après insertion, le nombre d’objets dans la boîte, les plus grand et plus petit objets déjà présents, et le ratio de remplissage de l’espace résiduel.

**Slide actuel : 27 — Architecture et données d'entraînement**

Le modèle est un `GradientBoostingClassifier` de scikit-learn, utilisé comme un classeur de paires objet-boîte. Les probabilités servent à classer les boîtes faisables, pas seulement à produire une classe binaire.

L’entraînement se fait par imitation de Best-Fit Decreasing. Quand BFD choisit une boîte, cette paire reçoit l’étiquette positive ; les autres boîtes faisables reçoivent l’étiquette négative.

Pour réduire le décalage entre entraînement et utilisation réelle, on entraîne le modèle avec deux types de traces : des traces BFD complètes, et des traces obtenues après destruction partielle d’une solution BFD.

---

## 6. Mohamed El Amine Kherroubi — Protocole expérimental et jeux de données

> **Slides couverts : 28 à 30.**

**Slide actuel : 28 — Séparation : évaluation expérimentale**

On passe maintenant à l’évaluation expérimentale : protocole, jeux de données, ablation et résultats.

**Slide actuel : 29 — Protocole expérimental**

Pour évaluer la méthode, on utilise le même protocole pour tous les tests.

La graine aléatoire est fixée à 42. ALNS utilise 300 itérations. La température initiale vaut 1 sur ln 2, et le coefficient de refroidissement est 0,9995.

Pour LinUCB, le paramètre d’exploration alpha vaut 0,3, avec 300 appels de phase de démarrage. Pour la réparation apprise, on utilise un seul modèle pré-entraîné sur tous les jeux de données.

La métrique principale est l’écart à la borne : nombre de boîtes utilisées moins LB1. Un écart de zéro veut dire que la solution atteint LB1 et qu’elle est donc certifiée optimale par cette borne.

**Slide actuel : 30 — Jeux de données**

On teste sur cinq familles de jeux de données de référence.

Scholl-2 contient des tailles plutôt uniformes. Falkenauer-T a une structure en triplets, ce qui rend LB1 moins fiable. Falkenauer-U contient aussi des tailles uniformes, mais avec une autre capacité. Wäscher est lié aux problèmes de découpe. Enfin, Hard28 contient des instances volontairement difficiles.

Cette diversité permet de vérifier si la méthode reste stable sur plusieurs structures d’instances, et pas seulement sur un cas particulier.

---

## 7. Rayan Boukakiou — Résultats : ablation et généralisation

> **Slides couverts : 31 à 32.**

**Slide actuel : 31 — Test 1 : ablation**

Le premier test est une étude d’ablation. Le but est de mesurer séparément l’effet de LinUCB et celui du modèle de réparation GBT.

Sur Scholl-2, avec cinq instances de 50 objets, la version sans apprentissage obtient un écart moyen de 0,20.

Quand on active seulement LinUCB, l’écart moyen passe à 0,00. C’est le meilleur résultat dans cette étude, avec un temps de calcul encore faible.

Quand on active seulement le modèle GBT, l’écart reste à 0,20, mais le temps augmente. Avec les deux composants ensemble, l’écart reste aussi à 0,20 sur ce petit échantillon.

La conclusion est donc que, dans ces tests, le choix adaptatif des opérateurs est le composant le plus utile. Le modèle GBT est bien intégré, mais son coût n’est pas encore compensé par un gain de qualité sur les petites instances.

**Slide actuel : 32 — Test 2 : généralisation multi-jeux**

Le deuxième test regarde la généralisation sur plusieurs familles.

La méthode combinée, c’est-à-dire LinUCB plus réparation GBT, reste strictement à moins d’une boîte de LB1 sur quatre familles sur cinq : Scholl-2, Falkenauer-U, Wäscher et Hard28.

Sur Falkenauer-T, l’écart moyen est exactement de 1,00. Le slide le signale séparément, car cette famille a une structure en triplets qui affaiblit LB1. Cet écart n’indique donc pas forcément une mauvaise recherche.

Globalement, les résultats montrent que la méthode est assez robuste, même sans réglage spécifique pour chaque famille.

---

## 8. Idris Yassine Ziadi — Résultats comparatifs

> **Slide couvert : 33.**

**Slide actuel : 33 — Test 3 : comparaison**

Le troisième test compare notre méthode avec plusieurs approches classiques.

On compare avec FFD et BFD, le recuit simulé, la recherche tabou, un algorithme génétique, l’optimisation par colonies de fourmis, et notre ALNS avec les deux composants d’apprentissage.

Sur Scholl-2 avec cinq instances de 50 objets, les méthodes constructives sont très rapides, mais elles gardent un écart moyen de 1,80.

Le recuit simulé et la recherche tabou ont aussi un écart de 1,80 dans ce test. L’algorithme génétique améliore le résultat avec un écart de 1,00, mais il prend plus de temps.

Notre ALNS dual-learning obtient un écart moyen de 0,20 en environ 0,36 seconde.

L’optimisation par colonies de fourmis obtient un écart de 0,00, donc une meilleure qualité sur ce test, mais avec un temps moyen d’environ 3,03 secondes. Elle est donc beaucoup plus lente que notre méthode.

Notre approche n’est pas toujours la meilleure en qualité pure, mais elle donne un très bon compromis entre qualité et temps de calcul. Dans ce tableau, aucune autre méthode n’obtient à la fois un écart plus faible et un temps plus faible que notre méthode, sauf ACO qui est meilleure en qualité mais beaucoup plus lente.

Il faut rester prudent, car les méthodes stochastiques sont évaluées sur une seule exécution. Avec plusieurs graines aléatoires, les classements pourraient changer.

---

## 9. Idris Himeur — Synthèse, limites et conclusion

> **Slides couverts : 34 à 36.**

**Slide actuel : 34 — Séparation : synthèse et conclusion**

On arrive maintenant à la synthèse, aux limites et aux perspectives.

**Slide actuel : 35 — Synthèse des résultats**

Pour résumer, on retient plusieurs points.

Premièrement, le choix adaptatif des opérateurs est le composant d’apprentissage le plus important dans nos tests. Dans l’étude d’ablation, LinUCB seul ferme l’écart moyen de 0,20 à 0,00.

Deuxièmement, le modèle de réparation GBT fonctionne, mais il n’apporte pas encore de gain visible sur les petites instances. Pour l’instant, il ajoute surtout du temps de calcul.

Troisièmement, la méthode combinée se généralise correctement sur plusieurs familles de jeux de données de référence : quatre familles sur cinq sont strictement sous une boîte d’écart moyen, et Falkenauer-T est isolée à 1,00 à cause de la faiblesse de LB1 sur les structures en triplets.

Quatrièmement, les deux composants d’apprentissage n’ont pas le même impact. C’est une information importante pour améliorer la suite du projet.

Enfin, par rapport aux métaheuristiques classiques testées, notre méthode offre un bon compromis entre qualité et vitesse.

Il faut garder les limites en tête : les tests utilisent seulement de petits échantillons de cinq instances, plusieurs comparaisons sont faites avec une seule exécution, et LB1 est une borne simple.

**Slide actuel : 36 — Conclusions**

En conclusion, notre contribution principale est une ALNS hybride pour le bin packing 1D, avec deux composants activables séparément : LinUCB pour choisir les opérateurs de destruction, et un modèle GBT supervisé pour guider la réparation.

La deuxième contribution est la chaîne d’apprentissage de la réparation : données augmentées, contrat de 11 caractéristiques, contrôles de qualité, et versionnement des caractéristiques pour éviter les incompatibilités silencieuses.

La troisième contribution est le bandit LinUCB avec phase de démarrage Thompson Sampling, puis récompense normalisée par l’écart à la borne.

Les prochaines étapes seraient de tester plus d’instances, de faire des moyennes sur plusieurs graines, d’utiliser une meilleure borne inférieure comme Martello-Toth L2, et d’améliorer le modèle de réparation, par exemple avec une réparation par apprentissage par renforcement de bout en bout.

---

## 10. Adem Abdelhafidh Diar — Mot de fin

> **Slide couvert : 36, après la conclusion technique.**

**Slide actuel : 36 — Mot de fin**

Pour finir, notre objectif n’était pas de remplacer les heuristiques classiques par un modèle opaque.

Nous avons plutôt gardé une méthode d’optimisation connue, ALNS, et ajouté de l’apprentissage seulement aux décisions répétées où cela peut aider.

Les résultats montrent que cette idée est prometteuse, surtout pour le choix adaptatif des opérateurs.

Merci pour votre attention. Nous sommes prêts à répondre à vos questions.
