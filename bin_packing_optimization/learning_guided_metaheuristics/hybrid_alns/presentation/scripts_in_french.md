# Scripts — Hybrid ALNS for the Bin Packing Problem

### Présentation 15 minutes

---

## 1. Mohamed El Amine Kherroubi — Introduction et plan

**Slide 1 — Titre**

Bonjour. Aujourd'hui on va vous présenter une métaheuristique hybride pour le problème de bin packing à une dimension, intitulée Hybrid ALNS.

---

**Slide 2 — Introduction**

Le bin packing est un problème d'optimisation combinatoire présent dans de nombreux domaines industriels, comme la découpe de matériaux, le chargement de camions ou encore l'allocation de ressources dans le cloud.

Le problème est NP-difficile. Les méthodes exactes permettent d'obtenir des solutions optimales, mais leur coût de calcul devient rapidement prohibitif. Les heuristiques classiques sont rapides, mais appliquent toujours les mêmes règles et se retrouvent souvent bloquées dans des optima locaux. Les métaheuristiques permettent généralement d'obtenir de meilleures solutions, mais les choix effectués pendant la recherche reposent le plus souvent sur des mécanismes génériques qui n'exploitent pas pleinement les informations accumulées au cours de l'exécution.

L'idée est donc d'intégrer du machine learning directement dans la métaheuristique afin d'exploiter ces informations. Nous proposons deux composants visant à guider la recherche. Comme ils sont indépendants, nous pouvons les évaluer séparément et mesurer précisément la contribution de chacun.

---

**Slide 3 — Plan**

La présentation va couvrir cinq parties : la définition du problème, une revue de littérature, notre solution, les tests et résultats, et enfin la synthèse et la conclusion.

---

## 2. Rayan Boukakiou — Définition du problème

**Slide 4 — Définition formelle**

Formellement, le bin packing est défini ainsi : on a n objets de tailles entières et des boîtes identiques de capacité C. Une solution réalisable est une partition de tous les objets en groupes telle que la somme des tailles dans chaque groupe ne dépasse pas C. L'objectif est de minimiser le nombre de groupes, donc le nombre de boîtes utilisées.

---

**Slide 5 — Borne inférieure LB1**

Pour évaluer nos solutions, on utilise une borne inférieure appelée LB1. C'est le plafond de la somme totale des tailles divisée par C. Toute solution doit stocker l'intégralité du volume et chaque boîte ne peut en contenir qu'au plus C, donc on ne peut pas faire moins que ça. Il existe des bornes plus précises, notamment celle de Martello et Toth, qui tient compte des gros objets qui ne peuvent pas coexister dans une même boîte. Mais LB1 nous suffit comme indicateur de progression dans ce projet. Quand notre solution atteint exactement LB1, on sait qu'elle est optimale.

---

## 3. Adem Abdelhafidh Diar — Revue de littérature

**Slide 6 — État de l'art des méthodes de résolution du BPP**

Les méthodes de résolution du bin packing se divisent en deux grandes familles. Du côté des méthodes exactes, les approches par branch-and-bound et par génération de colonnes permettent d'obtenir des solutions optimales, mais leur complexité les rend impraticables sur de grandes instances. Du côté des métaheuristiques, on trouve des approches classiques comme le recuit simulé, la recherche tabou, les algorithmes génétiques ou les colonies de fourmis. Ces méthodes offrent de bonnes solutions en temps raisonnable, mais reposent sur des mécanismes génériques qui n'adaptent pas leur comportement à l'instance en cours de résolution. LNS, et en particulier ALNS, se distingue en travaillant sur des voisinages larges et en choisissant adaptativement ses opérateurs, ce qui lui permet d'explorer plus efficacement l'espace de solutions.

---

**Slide 7 — Machine learning et métaheuristiques**

Le machine learning offre une façon rigoureuse de rendre ces décisions pilotées par les données. En suivant la taxonomie de Bengio et al., le ML peut être hybridé avec les métaheuristiques selon trois modes : premièrement, prédire la configuration ou les paramètres d'un algorithme hors ligne ; deuxièmement, guider les décisions de recherche en ligne à partir de l'état courant ; troisièmement, remplacer un opérateur artisanal par un modèle appris. Le premier mode s'adapte mal au cours d'une même exécution, et le troisième peut être coûteux en données et en calcul s'il est utilisé seul. Nous combinons donc les deuxième et troisième modes aux deux décisions qui influencent le plus la qualité d'ALNS.

On peut identifier trois directions de travaux connexes. La première est la sélection adaptative d'opérateurs, formalisée comme un problème de bandit. La deuxième est la réparation apprise par imitation d'une heuristique experte, ce qu'on fait exactement pour la réinsertion des objets. La troisième est l'utilisation du deep learning dans LNS, avec du reinforcement learning pour apprendre des opérateurs complets. Nous proposons quelque chose de plus léger et de plus interprétable : deux composants ciblés sur des décisions précises, sans remplacer toute la boucle d'optimisation.

---

## 4. Mohamed El Amine Kherroubi — Architecture globale et métaheuristique

**Slide 9 — Architecture globale**

Voilà comment notre approche fonctionne de bout en bout. On part de l'instance, BFD construit une solution initiale, et ensuite la boucle ALNS démarre. À chaque itération, LinUCB regarde l'état courant de la recherche et choisit un opérateur de destruction. Cet opérateur retire un sous-ensemble d'objets de leurs boîtes. Le modèle GBT évalue ensuite les boîtes disponibles pour guider la réinsertion de ces objets. La nouvelle solution est acceptée ou rejetée selon un critère de recuit simulé, et la récompense est renvoyée à LinUCB pour qu'il mette à jour ses estimations. Les deux composants ML sont complètement indépendants, on peut activer l'un sans l'autre.

---

**Slide 10 — Initialisation BFD**

Avant que la boucle démarre, BFD construit la solution initiale. Il trie les objets du plus grand au plus petit, et pour chaque objet il cherche la boîte ouverte qui laissera le moins d'espace libre après insertion. S'il n'y en a pas, il ouvre une nouvelle boîte. Ce point de départ compact est important parce que ça réduit d'emblée le travail qu'ALNS doit faire.

---

**Slide 12-13 — Pourquoi LNS**

La recherche locale classique se bloque dans des optima locaux dont elle ne peut pas sortir en déplaçant un seul objet à la fois. LNS résout ça en travaillant sur des voisinages beaucoup plus grands : on retire partiellement des objets de la solution et on les réinsère, ce qui permet d'atteindre des régions de l'espace de solutions inaccessibles autrement. ALNS va encore plus loin en choisissant adaptativement quel opérateur de destruction utiliser parmi plusieurs.

---

**Slide 14-15 — Opérateurs et rayon**

On a trois opérateurs de destruction. Le premier retire des objets au hasard pour explorer globalement. Le deuxième cible les boîtes les moins remplies, les parties les plus faibles de la solution. Le troisième choisit des objets de tailles similaires pour essayer de les recombiner différemment. Le nombre d'objets retirés est tiré aléatoirement entre 5 et 25 pourcent de n, et cette borne haute s'élargit automatiquement quand la recherche stagne pour forcer plus de diversification.

---

**Slide 17-18 — Acceptation et redémarrage**

Pour l'acceptation on utilise le critère du recuit simulé : une solution moins bonne peut quand même être acceptée avec une probabilité qui diminue au fil du temps, ça évite de rester coincé. Un mécanisme de réchauffage progressif se déclenche aussi en cas de stagnation prolongée pour éviter que la température ne s'effondre trop tôt. Si la stagnation atteint sa limite, un redémarrage remet la solution courante à la meilleure qu'on a vue, et réduit la fenêtre de patience pour que les prochains redémarrages arrivent plus vite.

---

## 5. Idriss Yassine Ziadi — Composant online : LinUCB

**Slide 20 — Vue d'ensemble des composants ML**

Notre solveur intègre deux composants d'apprentissage indépendants. Je vais vous parler du premier, qui choisit l'opérateur de destruction à chaque itération. Adem vous parlera ensuite du second, qui guide la réinsertion des objets.

---

**Slide 21-22 — LinUCB**

La sélection d'opérateur se passe en deux phases. Les 300 premiers appels utilisent Thompson Sampling : chaque opérateur a une distribution de probabilité, on tire un échantillon de chacun et on choisit celui qui a le score le plus élevé. Ça permet d'explorer correctement au début sans dépendre d'informations qu'on n'a pas encore.

Ensuite LinUCB prend le relais. Pour chaque opérateur, il calcule un score qui combine ce qu'il estime être la performance de cet opérateur dans le contexte actuel, et un bonus d'exploration qui est plus grand quand l'incertitude est forte. On choisit l'opérateur avec le score le plus élevé et on met à jour ses paramètres avec la récompense observée.

---

**Slide 23-24 — Vecteur de contexte et récompense**

Le contexte qu'on passe à LinUCB c'est un vecteur à 5 dimensions : la température relative, le niveau de stagnation, le rapport entre LB1 et le coût actuel, le rayon de destruction relatif, et l'avancement global dans le budget d'itérations. Tout est normalisé entre 0 et 1.

Pour la récompense, si on a économisé des boîtes, elle est proportionnelle au gain normalisé par l'écart à LB1 — ce qui fait qu'une amélioration quand on est déjà près de l'optimum est plus valorisée. Si la solution est acceptée sans gain elle vaut 0,2, et si elle est rejetée elle vaut zéro.

---

## 6. Adem Abdelhafidh Diar — Composant offline : GBT

**Slide 25-27 — Réparation GBT**

Quand on répare la solution après une destruction, on réinsère les objets du plus grand au plus petit. Pour chaque objet, le modèle GBT regarde toutes les boîtes où on pourrait le mettre et attribue un score à chacune. On choisit la boîte avec le score le plus élevé.

Le modèle a été entraîné à imiter BFD. Pour chaque paire objet-boîte, on lui apprend si c'est le choix que BFD aurait fait ou non. Chaque paire est représentée par 11 caractéristiques : la taille de l'objet, sa position dans le tri, la charge de la boîte, son résiduel, et quelques interactions entre les deux.

Pour les données d'entraînement on a deux types de traces. Des traces où BFD part de zéro, et des traces post-destruction où on simule exactement ce qui se passe dans ALNS — une solution partiellement détruite qu'on doit réparer. Ce deuxième type est important pour que le modèle soit utile en conditions réelles.

Pour compléter sur l'implémentation, le bundle entraîné inclut un numéro de version des caractéristiques. Si on change le format des features et qu'on charge un ancien modèle, ça lève une erreur immédiatement au lieu de produire des résultats silencieusement faux. C'est un détail technique mais qui évite des bugs difficiles à détecter.

---

## 7. Idris Himeur — Réglage des paramètres

**Slide 29 — Réglage des paramètres**

La graine aléatoire est fixée à 42 pour la reproductibilité. ALNS tourne sur 300 itérations avec une température initiale de 1 sur ln 2 et un coefficient de refroidissement de 0,9995. Pour le bandit, le paramètre d'exploration est 0,3 et la phase de démarrage Thompson Sampling dure 300 appels. La métrique principale est l'écart à LB1 : le nombre de boîtes utilisées moins LB1. Quand cet écart vaut zéro, la solution est certifiée optimale selon cette borne.

---

**Slide 30 — Jeux de données**

On a testé sur cinq familles de benchmarks avec des structures très différentes. Scholl-2 avec des tailles uniformes et entre 50 et 500 objets. Falkenauer-T avec une structure en triplets — attention, sur cette famille LB1 est systématiquement trop faible donc les écarts ne sont pas directement comparables aux autres. Falkenauer-U avec une capacité de 150. Wäscher issu de la découpe industrielle avec une capacité de 10 000. Et Hard28, conçu pour être difficile, avec entre 160 et 200 objets. Je vais maintenant présenter les résultats.

---

## 8. Idris Himeur — Tests 1 et 2

**Slide 31 — Test 1 : ablation**

Le premier test cherche à isoler la contribution de chaque composant. On active et désactive LinUCB et le GBT séparément sur 5 instances Scholl-2 à 50 objets.

Sans apprentissage du tout, l'écart moyen est 0,20. Avec LinUCB seul, l'écart tombe à zéro en 0,19 secondes. Avec le GBT seul, l'écart reste à 0,20 mais le temps double à 0,35 secondes. Avec les deux combinés, l'écart est toujours 0,20 en 0,36 secondes.

LinUCB est clairement le composant qui fait la différence : seul, il ferme complètement l'écart. Le GBT est bien intégré mais sur ces petites instances son coût en temps n'est pas compensé par un gain de qualité. Son bénéfice est attendu sur des instances plus grandes.

---

**Slide 32 — Test 2 : généralisation**

Le deuxième test vérifie que la méthode se comporte bien sur des familles très différentes, sans aucun réglage spécifique par famille. Scholl-2 et Falkenauer-U arrivent à 0,20, Wäscher à 0,60, Hard28 à 0,67. Falkenauer-T est à 1,00, mais comme on l'a dit c'est probablement dû à la faiblesse de LB1 sur cette famille et pas à une mauvaise qualité de recherche. En tout cas, 4 familles sur 5 sont à moins d'une boîte de LB1 sans aucun réglage, ce qui valide la robustesse de l'approche.

---

## 9. Idris Himeur — Test 3 : comparatif

**Slide 33 — Test 3 : comparaison**

Le troisième test situe notre méthode par rapport aux approches classiques. On compare sur les mêmes 5 instances Scholl-2. FFD, BFD, recuit simulé et recherche tabou sont tous à un écart de 1,80. L'algorithme génétique descend à 1,00 en 0,82 secondes. Notre méthode atteint 0,20 en 0,36 secondes. La colonie de fourmis atteint 0,00 mais elle prend 3,03 secondes, soit environ 8 fois plus que nous.

On n'est pas les meilleurs en qualité pure, mais on offre le meilleur compromis qualité-vitesse du comparatif. Aucune autre méthode ne fait à la fois un écart plus faible et un temps plus court que les nôtres.

---

## 10. Rayan Boukakiou — Synthèse et conclusion

**Slide 35 — Synthèse**

Avant de conclure, voilà ce que les trois tests nous ont appris.

Premier constat : LinUCB est le composant dominant. Sur les petites instances, il suffit à lui seul à atteindre LB1, ce que la combinaison complète ne fait pas mieux.

Deuxième constat : le GBT fonctionne et il est bien intégré, mais son apport n'est pas encore visible à petite échelle où il ajoute surtout du temps de calcul. Il faudrait le tester sur des instances plus grandes pour quantifier son vrai bénéfice.

Troisième constat : les deux composants ne contribuent pas de la même façon. Il y a un déséquilibre clair que l'ablation révèle, et c'est un axe d'amélioration évident pour la suite.

Quatrième constat : malgré tout, la méthode combinée se généralise sur 4 familles sur 5 et elle domine les métaheuristiques classiques sur le plan qualité-vitesse.

Il faut garder en tête les limites : on évalue sur des tranches de 5 instances, les méthodes stochastiques sont comparées sur une seule exécution, et on utilise uniquement LB1 qui est plus faible que la borne de Martello-Toth.

---

**Slide 36 — Conclusions**

Pour conclure, notre contribution principale est un ALNS hybride pour le bin packing avec deux composants d'apprentissage activables séparément. Le premier est LinUCB avec un démarrage Thompson Sampling et une récompense normalisée par l'écart à LB1. Le second est un modèle de réparation supervisé entraîné par imitation de BFD, avec des données augmentées par des traces post-destruction.

Les suites naturelles seraient d'évaluer sur plus d'instances en moyennant sur plusieurs graines, d'utiliser la borne de Martello-Toth à la place de LB1, et d'explorer une réparation par renforcement de bout en bout pour ne plus dépendre du supervisé.

Merci pour votre attention, on est disponibles pour vos questions.
