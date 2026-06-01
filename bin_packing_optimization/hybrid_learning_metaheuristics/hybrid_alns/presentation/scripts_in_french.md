# Scripts de présentation — Hybrid ALNS for the Bin Packing Problem

---

## 1. Mohamed El Amine Kherroubi — Introduction et plan

> **Slides couverts : 1 à 3.**

---

**Slide 1 — Titre**

Bonjour à tous. Nous allons vous présenter aujourd'hui notre projet : une métaheuristique hybride pour le problème de bin packing à une dimension.

Le titre de notre solution est Hybrid ALNS. Nous expliquerons en détail ce que cela signifie au fil de la présentation.

---

**Slide 2 — Introduction**

Le problème du bin packing à une dimension est omniprésent dans l'industrie : découpe de matériaux, chargement de véhicules, allocation de ressources dans le cloud. Dans tous ces contextes, on cherche à placer des objets de tailles variées dans le minimum de conteneurs de capacité fixe.

Ce problème est NP-difficile au sens fort. Les méthodes exactes trouvent la solution optimale, mais leur temps de calcul explose dès que le nombre d'objets augmente. Les heuristiques classiques, à l'inverse, sont rapides mais reposent sur des règles de décision fixes qui ne s'adaptent pas à la structure du problème en cours de résolution. Elles se retrouvent ainsi fréquemment bloquées dans des optima locaux.

C'est précisément là qu'intervient notre motivation pour une approche hybride. L'idée centrale est la suivante : intégrer de l'apprentissage automatique à l'intérieur d'une métaheuristique, à des points de décision précis, sans remplacer la boucle d'optimisation.

Nous avons ajouté deux composants d'apprentissage indépendants à un cadre de type ALNS, c'est-à-dire Adaptive Large-Neighborhood Search. Le premier composant apprend, en ligne, quel opérateur de destruction choisir en fonction de l'état courant de la recherche. Le second apprend, hors ligne, dans quelle boîte replacer chaque objet déplacé, en imitant le comportement d'une heuristique de référence.

Ces deux composants sont activables séparément, ce qui permet une étude d'ablation rigoureuse.

---

**Slide 3 — Plan**

La présentation est organisée en six parties. Nous commençons par la définition formelle du problème, puis la revue de littérature sur l'hybridation entre métaheuristiques et apprentissage automatique. Nous présentons ensuite notre solution dans le détail : architecture globale, cadre algorithmique, puis chacun des deux composants d'apprentissage. Nous enchaînons avec les tests et résultats, suivis d'une synthèse et d'une conclusion.

Je laisse maintenant la parole à Rayan, qui va définir formellement le problème.

---

## 2. Rayan Boukakiou — Définition du problème et stratégie algorithmique

> **Slides couverts : 4 à 6.**

---

**Slide 4 — Définition formelle**

Définissons le problème formellement. On dispose d'un ensemble de n objets, chacun ayant une taille entière positive notée s indice i, et d'un ensemble de boîtes identiques de capacité entière C. On suppose que chaque objet tient dans une boîte seule, c'est-à-dire que s indice i est inférieur ou égal à C pour tout i.

Une solution faisable est une partition de l'ensemble des objets en groupes B un, B deux, jusqu'à B m, telle que pour chaque groupe, la somme des tailles des objets qu'il contient ne dépasse pas C.

L'objectif est de minimiser m, le nombre de boîtes utilisées.

---

**Slide 5 — Borne inférieure LB un**

Pour évaluer la qualité d'une solution, nous utilisons une borne inférieure notée LB un. Elle est définie comme le plafond de la somme des tailles divisée par la capacité C. La preuve est immédiate : toute solution faisable doit contenir la totalité du volume des objets, et chaque boîte ne peut en stocker qu'au plus C unités. Le nombre de boîtes est donc au moins égal à ce quotient, arrondi au supérieur.

Un écart nul entre le nombre de boîtes utilisées et LB un signifie que la solution est optimale selon cette borne. C'est le critère de convergence que nous utilisons tout au long de nos tests.

Il existe des bornes plus précises, comme la borne L deux de Martello et Toth, qui tient compte des gros objets ne pouvant pas coexister dans une même boîte. Mais LB un suffit comme indicateur de progression dans le cadre de ce projet. Nous y reviendrons dans la section résultats, notamment pour la famille Falkenauer-T où cette borne est systématiquement trop faible.

---

**Slide 6 — Complexité et stratégie algorithmique**

Le bin packing à une dimension est NP-difficile au sens fort, par réduction depuis le problème trois-partition. Face à cette difficulté, trois grandes familles d'approches existent.

Les méthodes exactes, comme branch-and-bound ou branch-and-price, garantissent l'optimalité mais ne passent pas à l'échelle au-delà de quelques centaines d'objets. Les heuristiques d'approximation, comme FFD et BFD, sont très rapides et garantissent un rapport d'approximation borné, mais leur qualité est limitée par leurs règles fixes. Les métaheuristiques, enfin, n'offrent aucune garantie théorique mais explorent l'espace de solutions de façon adaptative.

Notre stratégie combine ces approches : on utilise BFD comme initialisation déterministe rapide, puis ALNS pour s'échapper des optima locaux et se rapprocher de LB un. Je laisse la parole à Idris Yassine, qui va présenter la littérature et l'architecture globale de notre solution.

---

## 3. Idris Yassine Ziadi — Littérature, architecture globale et initialisation

> **Slides couverts : 7 à 10.**

---

**Slide 7 — Revue de littérature**

Notre travail s'inscrit dans un courant précis : l'utilisation de l'apprentissage automatique à l'intérieur des métaheuristiques. Il est important de distinguer ce courant de son inverse, qui consiste à utiliser des métaheuristiques pour optimiser des modèles d'apprentissage — ce n'est pas notre sujet ici.

Trois axes de la littérature nous concernent directement.

Le premier est la sélection adaptative d'opérateurs. Plutôt que d'attribuer des probabilités fixes à chaque opérateur de recherche, on apprend en ligne lesquels sont les plus efficaces. Fialho et collaborateurs ont formalisé cette idée comme un problème de bandit en deux mille dix. Des travaux ultérieurs l'ont intégrée dans des cadres évolutionnaires. Nous utilisons LinUCB, proposé par Chu et collaborateurs en deux mille onze, qui est une extension contextuelle : le choix de l'opérateur dépend de l'état courant de la recherche, et non seulement de récompenses passées.

Le deuxième axe est la réparation apprise. Khalil et collaborateurs ont montré en deux mille dix-sept que des réseaux de neurones peuvent apprendre des politiques de construction compétitives avec les heuristiques classiques sur des problèmes combinatoires. Notre approche s'appuie sur le clonage comportemental : on entraîne un modèle à imiter les décisions de BFD à partir de démonstrations, sans signal de récompense.

Le troisième axe est l'apprentissage dans LNS, Large-Neighborhood Search. Des travaux comme ceux de Hottung et Tierney en deux mille vingt, ou de Lu et collaborateurs en deux mille vingt-et-un, utilisent l'apprentissage par renforcement profond pour apprendre des opérateurs de destruction et réparation complets. Notre approche est volontairement plus légère et plus interprétable : nous ajoutons deux aides ciblées à des décisions précises, sans remplacer la métaheuristique par un modèle de bout en bout.

Notre contribution se situe à l'intersection des deux premiers axes : un ALNS avec double apprentissage, combinant un bandit contextuel en ligne pour la sélection d'opérateurs, et un réparateur supervisé hors ligne.

---

**Slide 9 — Architecture globale**

Voici l'architecture complète de notre solveur. On part d'une instance du problème, c'est-à-dire n objets avec leurs tailles et une capacité C. L'initialisation est réalisée par BFD, qui produit une solution de départ compacte.

La boucle principale est celle d'ALNS. À chaque itération, le bandit LinUCB choisit un opérateur de destruction en fonction d'un vecteur de contexte à cinq dimensions décrivant l'état de la recherche. L'opérateur sélectionné retire un sous-ensemble d'objets de la solution courante. Le modèle de réparation, un classifieur à gradient boosting, note ensuite les boîtes faisables pour guider la réinsertion de chaque objet déplacé. La solution reconstruite est évaluée par un critère d'acceptation de type recuit simulé. La meilleure solution rencontrée est conservée tout au long de la recherche. Enfin, la récompense observée est renvoyée à LinUCB pour mettre à jour ses estimations.

Les deux composants d'apprentissage sont indépendants et activables séparément, ce qui est indispensable pour les tester en ablation.

---

**Slide 10 — Initialisation BFD**

Avant que la boucle ALNS démarre, BFD construit la solution initiale. L'algorithme trie d'abord les objets du plus grand au plus petit. Pour chaque objet, il évalue toutes les boîtes ouvertes qui peuvent l'accueillir et place l'objet dans celle qui laissera le moins d'espace libre après insertion. Si aucune boîte existante ne convient, une nouvelle boîte est ouverte.

Ce point de départ compact est important : ALNS aura moins de boîtes à améliorer dès le départ, ce qui accélère la convergence vers LB un. Je laisse maintenant la parole à Idris Himeur, qui va présenter le cadre ALNS en détail.

---

## 4. Idris Himeur — ALNS, destruction et acceptation

> **Slides couverts : 11 à 18.**

---

**Slide 11 — Séparateur : cadre métaheuristique ALNS**

*(slide de transition — aucun script oral nécessaire)*

---

**Slide 12 — Pourquoi la recherche locale seule échoue**

La recherche locale classique explore le voisinage de la solution courante en effectuant de petites modifications, par exemple déplacer un seul objet d'une boîte à une autre. Cette approche est rapide, mais elle converge vers des optima locaux : des solutions dont aucune modification marginale n'améliore l'objectif, même si elles sont loin de l'optimum global.

Dans le bin packing, ces optima sont particulièrement denses car la qualité d'une solution dépend fortement des groupements d'objets au sein de chaque boîte. Aucune séquence bornée de déplacements unitaires ne peut de façon fiable s'en échapper.

La réponse à ce problème est la Large-Neighborhood Search, proposée par Shaw en mille neuf cent quatre-vingt-dix-huit. L'idée est d'opérer sur des voisinages implicitement exponentiels en détruisant partiellement la solution courante, puis en la réparant.

---

**Slide 13 — L'itération LNS**

Chaque itération de LNS comporte deux phases.

La phase de destruction retire un sous-ensemble d'objets de leurs boîtes, produisant une solution partielle. La phase de réparation réinsère tous ces objets pour restaurer la faisabilité et obtenir une nouvelle solution complète.

Le voisinage exploré est implicitement exponentiel en fonction du nombre d'objets déplacés : la réparation peut produire n'importe quelle complétion faisable de la solution partielle, permettant d'atteindre des régions de l'espace de solutions inaccessibles par de petits déplacements.

ALNS, proposé par Ropke et Pisinger en deux mille six, étend LNS en sélectionnant adaptativement l'opérateur de destruction parmi un portefeuille d'opérateurs. C'est ce mécanisme de sélection que notre bandit LinUCB vient améliorer.

---

**Slide 14 — Opérateurs de destruction**

Trois opérateurs de destruction sont disponibles dans notre implémentation.

L'opérateur aléatoire retire exactement k objets choisis uniformément au hasard parmi tous les objets placés. Il favorise l'exploration globale.

L'opérateur worst-load trie les boîtes par charge croissante et retire des objets en commençant par les boîtes les moins remplies. Il cible les parties les plus faibles de la solution courante.

L'opérateur related-item choisit un objet graine uniformément au hasard, puis retire les k moins un objets dont la taille est la plus proche de celle de la graine. Il permet de recombiner des groupes d'objets de tailles similaires.

Les trois opérateurs garantissent qu'au moins un objet reste placé à chaque itération.

---

**Slide 15 — Rayon de destruction**

Le nombre d'objets retirés à chaque itération, noté k, est tiré uniformément dans un intervalle dont les bornes dépendent de n. La borne inférieure est cinq pour cent de n et la borne supérieure est vingt-cinq pour cent de n.

Lorsque la recherche stagne, c'est-à-dire que le nombre d'itérations sans amélioration augmente, la borne supérieure de cet intervalle s'élargit progressivement. Cela force une diversification plus forte pour sortir du bassin d'attraction de l'optimum local courant.

---

**Slide 16 — Séparateur : mécanisme d'acceptation**

*(slide de transition — aucun script oral nécessaire)*

---

**Slide 17 — Acceptation et refroidissement**

Une solution candidate est acceptée si elle est meilleure ou égale à la solution courante. Si elle est moins bonne, elle peut tout de même être acceptée avec une probabilité qui décroît avec la température, selon le critère du recuit simulé classique.

Un refroidissement géométrique est appliqué après chaque itération : la température est multipliée par un coefficient alpha légèrement inférieur à un, valant zéro virgule neuf neuf neuf cinq dans nos expériences.

Pour éviter un refroidissement prématuré lors de stagnations prolongées, un mécanisme de réchauffage progressif est activé : si la stagnation dépasse un certain seuil multiple de la limite de patience, la température est remontée à au moins trente-cinq pour cent de la température initiale.

---

**Slide 18 — Redémarrage de diversification**

Lorsque la stagnation atteint la limite de patience maximale, un redémarrage est déclenché. La solution courante est réinitialisée à la meilleure solution trouvée jusqu'ici. La température est réchauffée à au moins vingt pour cent de la température initiale. La fenêtre de patience est réduite de un tiers, de sorte que les redémarrages suivants se déclenchent plus tôt.

Le seul critère d'arrêt dur reste le nombre maximum d'itérations. Le mécanisme de redémarrage ne termine jamais la recherche prématurément.

Je passe maintenant la parole à Adem, qui va présenter en détail nos deux composants d'apprentissage automatique.

---

## 5. Adem Abdelhafidh Diar — Composants d'apprentissage automatique

> **Slides couverts : 19 à 27.**

---

**Slide 19 — Séparateur : composants d'apprentissage automatique**

*(slide de transition — aucun script oral nécessaire)*

---

**Slide 20 — Deux composants indépendants**

Notre solveur intègre deux composants d'apprentissage automatique. Le premier décide quel opérateur de destruction appliquer à chaque itération. Le second décide dans quelle boîte réinsérer chaque objet déplacé lors de la réparation. Ils sont indépendants dans leur implémentation et peuvent être activés ou désactivés séparément. C'est cette indépendance qui rend l'étude d'ablation possible et rigoureuse.

---

**Slide 21 — Composant un, phase un : Thompson Sampling**

La sélection d'opérateur s'effectue en deux phases.

Pendant les trois cents premiers appels, on utilise le Thompson Sampling bêta-bernoulli. Chaque opérateur démarre avec une distribution bêta uniforme, c'est-à-dire bêta un virgule un. À chaque appel, on tire un échantillon depuis la distribution de chaque opérateur, on choisit celui dont l'échantillon est le plus élevé, puis on met à jour uniquement la distribution de l'opérateur sélectionné en fonction de la récompense observée.

Cette phase de démarrage permet d'identifier les opérateurs les plus performants sans dépendre du vecteur de contexte, qui serait mal estimé en tout début de recherche. Elle atténue ainsi le problème de démarrage à froid inhérent à LinUCB.

---

**Slide 22 — Composant un, phase deux : LinUCB**

À partir du trois-cent-unième appel, LinUCB prend le relais. Pour chaque opérateur k, le score est la somme de la performance estimée et d'un terme d'exploration pondéré par alpha, qui vaut zéro virgule trois. La performance estimée est le produit scalaire entre les paramètres appris de l'opérateur et le vecteur de contexte courant. Le terme d'exploration est proportionnel à l'incertitude sur cette estimation.

L'opérateur dont le score est le plus élevé est sélectionné. Ses paramètres sont ensuite mis à jour selon une formule de rang un de Sherman-Morrison, ce qui est efficace en temps quadratique en la dimension du contexte.

---

**Slide 23 — Vecteur de contexte**

Le vecteur de contexte est à cinq dimensions, toutes normalisées entre zéro et un.

La première est la température relative : température courante divisée par la température initiale. Elle vaut un au début de la recherche et décroît vers zéro.

La deuxième est la progression de la stagnation : nombre d'itérations sans amélioration divisé par la limite de patience courante.

La troisième est le rapport LB un sur le coût de la solution courante. Elle vaut un lorsque la solution est optimale selon cette borne.

La quatrième est le rayon de destruction relatif : le nombre d'objets déplacés k divisé par n.

La cinquième est l'avancement global dans le budget d'itérations.

Cette normalisation garantit que le terme d'exploration de LinUCB est comparable d'une dimension à l'autre, sans normalisation supplémentaire.

---

**Slide 24 — Signal de récompense**

La récompense renvoyée à LinUCB après chaque itération dépend du résultat observé.

Si des boîtes ont été économisées, c'est-à-dire si la solution candidate est meilleure que l'incumbent, la récompense est proportionnelle au gain normalisé par l'écart courant à LB un, plafonnée à un. Cette normalisation est importante : elle valorise davantage un gain obtenu quand on est déjà proche de LB un, reflétant la difficulté croissante des améliorations.

Si la solution est acceptée sans économie de boîtes, la récompense vaut zéro virgule deux.

Si la solution est rejetée, la récompense est nulle.

---

**Slide 25 — Composant deux : réparation apprise, vue d'ensemble**

Après une destruction, chaque objet déplacé doit être réinséré. Les objets sont traités du plus grand au plus petit.

Pour chaque objet, on identifie l'ensemble des boîtes faisables, c'est-à-dire les boîtes dont la charge actuelle plus la taille de l'objet ne dépasse pas la capacité C. Si cet ensemble est vide, une nouvelle boîte est ouverte. Sinon, le modèle attribue un score à chaque boîte faisable et l'objet est placé dans la boîte au score le plus élevé.

L'objectif d'entraînement est un classifieur binaire sur des paires objet-boîte faisables : la boîte que BFD aurait choisie reçoit l'étiquette positive, toutes les autres reçoivent l'étiquette négative. Il s'agit d'un clonage comportemental depuis l'expert BFD.

---

**Slide 26 — Représentation des caractéristiques**

Chaque paire objet-boîte faisable est représentée par un vecteur à onze dimensions, toutes normalisées par C ou par n.

Les quatre premières décrivent l'objet : sa taille normalisée, sa taille normalisée au carré, son rang parmi les n objets du problème, et la fraction d'objets déplacés non encore réinsérés.

Les quatre suivantes décrivent la boîte candidate : sa charge normalisée, sa capacité résiduelle normalisée, le slack résiduel après insertion de l'objet, et le nombre d'objets déjà présents normalisé par n.

Les trois dernières capturent les interactions : la taille normalisée du plus grand objet déjà dans la boîte, celle du plus petit, et le ratio de remplissage résiduel, c'est-à-dire la fraction de la capacité résiduelle que cet objet consommerait.

---

**Slide 27 — Architecture du modèle et données d'entraînement**

Le modèle est un GradientBoostingClassifier de scikit-learn. Son inférence utilise les probabilités de classe pour classer les boîtes faisables, et non des décisions binaires dures.

Pour l'efficacité en inférence, on utilise un prédicteur rapide qui applique le scaler NumPy directement et appelle la fonction de prédiction brute du modèle, court-circuitant les validations par appel de scikit-learn. Le bundle sérialisé stocke le numéro de version des caractéristiques et leur nombre, et lève une erreur au chargement en cas de désaccord — ce qui prévient les incompatibilités silencieuses.

Les données d'entraînement combinent deux types de traces pour réduire le décalage entre distribution d'entraînement et distribution d'inférence. Les traces BFD complètes rejouent BFD depuis zéro sur des instances du benchmark. Les traces post-destruction construisent une solution BFD, évincent une fraction aléatoire de boîtes, puis relancent la réinsertion des objets déplacés avec le même étiquetage. Ce second type de traces simule des situations de réparation réelles telles qu'elles se produisent dans la boucle ALNS.

Je laisse maintenant la parole à Amine, qui va présenter le protocole expérimental.

---

## 6. Mohamed El Amine Kherroubi — Protocole expérimental et jeux de données

> **Slides couverts : 28 à 30.**

---

**Slide 28 — Séparateur : évaluation expérimentale**

*(slide de transition — aucun script oral nécessaire)*

---

**Slide 29 — Protocole expérimental**

Avant de présenter les résultats, décrivons le protocole. La graine aléatoire est fixée à quarante-deux pour la reproductibilité. ALNS tourne sur trois cents itérations. La température initiale vaut un sur le logarithme de deux, ce qui correspond au critère classique d'acceptation à cinquante pour cent d'une solution à un de plus que l'actuelle au départ. Le coefficient de refroidissement est zéro virgule neuf neuf neuf cinq.

Pour le bandit, alpha LinUCB vaut zéro virgule trois et la phase de démarrage Thompson Sampling dure trois cents appels. Un seul modèle GBT est pré-entraîné une fois pour toutes, sur des traces issues de l'ensemble des familles de benchmark.

La métrique principale est l'écart à LB un : nombre de boîtes utilisées moins LB un. Un écart nul signifie optimalité certifiée par cette borne.

L'environnement d'exécution est Python trois virgule treize, scikit-learn, sous Windows onze, sur un processeur Intel i neuf treize mille neuf cent cinquante HX avec soixante-quatre gigaoctets de mémoire vive.

---

**Slide 30 — Jeux de données**

Cinq familles de benchmark sont utilisées, couvrant des structures très différentes.

Scholl deux a une capacité de mille et entre cinquante et cinq cents objets, avec des tailles uniformes. Falkenauer-T a une capacité de mille et présente une structure en triplets, ce qui affaiblit systématiquement LB un par rapport à l'optimum réel — les écarts rapportés pour cette famille ne sont donc pas directement comparables aux autres. Falkenauer-U a une capacité de cent cinquante et des tailles uniformes, mais dans un régime de capacité différent. Wäscher a une capacité de dix mille et est issu de la découpe industrielle. Hard28 a une capacité de mille, entre cent soixante et deux cents objets, et est conçu pour être adversarialement difficile.

Cette diversité est importante : elle permet de vérifier que notre méthode se comporte correctement sur des structures très différentes, sans réglage spécifique par famille.

---

## 7. Rayan Boukakiou — Résultats : ablation et généralisation

> **Slides couverts : 31 à 32.**

---

**Slide 31 — Test un : étude d'ablation**

Le premier test isole la contribution individuelle de chaque composant d'apprentissage. On active et désactive LinUCB et le modèle GBT séparément, sur cinq instances de Scholl deux à cinquante objets.

La configuration sans apprentissage obtient un écart moyen de zéro virgule vingt. LinUCB seul ramène cet écart à zéro, avec un temps de calcul légèrement supérieur, zéro virgule dix-neuf secondes contre zéro virgule quinze. Le modèle GBT seul maintient l'écart à zéro virgule vingt mais double le temps de calcul, à zéro virgule trente-cinq secondes. La combinaison des deux donne également un écart de zéro virgule vingt, pour un temps de zéro virgule trente-six secondes.

La conclusion est claire : sur ces petites instances, LinUCB est le composant décisif — seul, il ferme complètement l'écart. Le modèle GBT est bien intégré, mais son coût en temps n'est pas encore compensé par un gain de qualité à cette échelle. Son bénéfice est attendu sur des instances plus grandes, ce qui reste une question empirique ouverte.

---

**Slide 32 — Test deux : généralisation multi-familles**

Le deuxième test évalue la généralisation de la méthode combinée sur l'ensemble des cinq familles de benchmark, sans aucun réglage spécifique par famille.

Quatre familles sur cinq obtiennent un écart strictement inférieur à une boîte : zéro virgule vingt pour Scholl deux, zéro virgule vingt pour Falkenauer-U, zéro virgule soixante pour Wäscher, et zéro virgule soixante-sept pour Hard28. La famille Falkenauer-T présente un écart de un virgule zéro, mais ce résultat est isolé pour une raison structurelle déjà signalée : sa structure en triplets affaiblit LB un, si bien que cet écart reflète probablement la faiblesse de la borne et non une mauvaise qualité de recherche.

Globalement, la méthode se généralise sans réglage spécifique par famille, ce qui valide la robustesse de l'approche.

---

## 8. Idris Yassine Ziadi — Résultats comparatifs

> **Slide couvert : 33.**

---

**Slide 33 — Test trois : étude comparative**

Le troisième test compare notre ALNS à double apprentissage à un ensemble représentatif de méthodes classiques, sur les mêmes cinq instances de Scholl deux à cinquante objets.

Les méthodes constructives, FFD et BFD, ont un écart de un virgule quatre-vingt pour un temps négligeable. Le recuit simulé et la recherche tabou ont également un écart de un virgule quatre-vingt, malgré un temps de calcul non nul. L'algorithme génétique descend à un virgule zéro, pour zéro virgule quatre-vingt-deux secondes. Notre ALNS à double apprentissage atteint zéro virgule vingt en zéro virgule trente-six secondes. L'optimisation par colonies de fourmis obtient zéro virgule zéro, mais nécessite en moyenne trois virgule zéro trois secondes, soit environ huit fois plus que notre méthode.

Notre méthode n'est pas la meilleure en qualité pure — la colonie de fourmis est plus précise — mais elle offre le meilleur compromis qualité-vitesse parmi toutes les alternatives testées. Aucune autre méthode ne combine à la fois un écart plus faible et un temps plus court que le nôtre.

Il faut noter que les méthodes stochastiques sont évaluées sur une seule exécution. En moyennant sur plusieurs graines, les classements pourraient évoluer.

---

## 9. Idris Himeur — Synthèse, limites et conclusion

> **Slides couverts : 34 à 36.**

---

**Slide 34 — Séparateur : synthèse et conclusion**

*(slide de transition — aucun script oral nécessaire)*

---

**Slide 35 — Synthèse des résultats**

Avant de conclure, faisons la synthèse de ce que les trois tests nous ont appris.

Premier constat : LinUCB est le composant d'apprentissage le plus impactant. Seul, il ferme l'écart de zéro virgule vingt à zéro sur les petites instances de l'ablation.

Deuxième constat : le modèle GBT est correctement implémenté et intégré, mais n'apporte pas encore de gain visible à petite échelle, où il ajoute principalement du temps de calcul. Son bénéfice reste à quantifier sur des instances plus grandes.

Troisième constat : les deux composants contribuent de façon asymétrique. L'ablation révèle un déséquilibre clair, qui constitue un axe d'amélioration évident.

Quatrième constat : la méthode combinée se généralise bien sur quatre familles sur cinq, et se positionne favorablement face aux métaheuristiques classiques sur le plan qualité-vitesse.

Les limites principales à garder en tête sont les suivantes : toutes les conclusions sont fondées sur des tranches d'évaluation de cinq instances, les méthodes stochastiques sont comparées sur une seule exécution, et nous utilisons uniquement la borne LB un, qui est plus faible que la borne de Martello-Toth.

---

**Slide 36 — Conclusions**

Notre contribution principale est une ALNS hybride pour le bin packing à une dimension, avec deux composants d'apprentissage activables séparément.

La première contribution secondaire est le bandit LinUCB avec phase de démarrage Thompson Sampling et signal de récompense normalisé par l'écart à LB un. La deuxième est la chaîne de réparation supervisée : augmentation des données par traces post-destruction, contrat à onze caractéristiques, contrôles de qualité par ROC-AUC, et versionnement pour prévenir les incompatibilités silencieuses.

Les suites naturelles à ce travail incluent une évaluation sur un plus grand nombre d'instances avec moyenne sur plusieurs graines, l'adoption d'une borne plus serrée comme Martello-Toth L deux, et l'exploration d'une réparation par renforcement de bout en bout pour supprimer la dépendance au supervisé.

---

## 10. Adem Abdelhafidh Diar — Mot de fin

> **Slide couvert : 36, après la conclusion technique.**

---

**Slide 36 — Mot de fin**

Notre démarche a été de rester au plus près de la métaheuristique : nous n'avons pas cherché à remplacer ALNS par un modèle opaque, mais à l'enrichir aux deux décisions répétées où l'apprentissage peut apporter une valeur ajoutée concrète — le choix de l'opérateur et le placement de réinsertion.

Les résultats montrent que cette approche est prometteuse, en particulier pour la sélection adaptative des opérateurs. La réparation apprise ouvre une piste de recherche intéressante, notamment à plus grande échelle.

Merci pour votre attention. Nous sommes disponibles pour vos questions.
