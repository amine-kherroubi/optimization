# Scripts de présentation — Hybrid ALNS

---

## 1. Mohamed El Amine Kherroubi — Introduction et plan

> **Slides couverts : 1 à 3.**

**Slide actuel : 1 — Titre**

Bonjour à tous. Nous présentons aujourd'hui Hybrid ALNS, notre solution au problème de bin packing 1D : comment placer des objets de tailles variées dans le minimum de boîtes de capacité fixe.

Notre méthode combine une métaheuristique ALNS, une sélection adaptative d'opérateurs via LinUCB, et une réparation guidée par un modèle supervisé.

**Slide actuel : 2 — Introduction**

Le bin packing est NP-difficile. Les méthodes exactes trouvent l'optimum mais ne passent pas à l'échelle. Les heuristiques classiques sont rapides, mais leurs règles fixes les laissent souvent piégées dans un optimum local.

Notre approche intègre de l'apprentissage automatique à deux points précis d'une métaheuristique existante, sans la remplacer :

- **en ligne**, LinUCB apprend quel opérateur de destruction choisir selon l'état courant de la recherche ;
- **hors ligne**, un modèle supervisé apprend dans quelle boîte replacer chaque objet déplacé, en imitant Best-Fit Decreasing.

Ces deux composants sont activables séparément, ce qui permet une étude d'ablation propre.

**Slide actuel : 3 — Plan**

La présentation suit le plan affiché : définition du problème, revue de littérature, solution proposée, expériences, synthèse et limites.

---

## 2. Rayan Boukakiou — Définition du problème et stratégie algorithmique

> **Slides couverts : 4 à 6.**

**Slide actuel : 4 — Définition formelle**

On a un ensemble d'objets à tailles entières positives et des boîtes de capacité entière uniforme. Une solution est faisable si chaque objet est placé une seule fois sans dépasser la capacité. L'objectif est de minimiser le nombre de boîtes utilisées.

**Slide actuel : 5 — Borne inférieure LB1**

La borne LB1 est la somme des tailles divisée par la capacité, arrondie au supérieur. Elle donne le minimum théorique sous utilisation parfaite du volume. Elle n'est pas toujours serrée — sur Falkenauer-T notamment, elle sous-estime systématiquement l'optimum réel — mais c'est le critère de convergence que nous utilisons : un gap nul signifie optimalité certifiée par LB1.

Des bornes plus fortes comme Martello-Toth L2 existent, mais LB1 suffit comme indicateur de progression pour ce projet.

**Slide actuel : 6 — Complexité et stratégie**

Face à la NP-difficulté du problème, notre stratégie est simple : construire une solution initiale avec Best-Fit Decreasing, puis utiliser ALNS pour s'échapper des optima locaux et se rapprocher de LB1.

---

## 3. Idris Yassine Ziadi — Littérature, architecture globale et initialisation

> **Slides couverts : 7 à 10.**

**Slide actuel : 7 — Revue de littérature**

Notre méthode s'inscrit dans trois familles de travaux.

La **sélection adaptative d'opérateurs** : au lieu de probabilités fixes, la méthode apprend pendant l'exécution quels opérateurs sont utiles. Nous l'implémentons avec LinUCB, un bandit contextuel. À chaque étape, on choisit un opérateur, on observe une récompense, et on met à jour les estimations.

La **réparation apprise** : un modèle entraîné à imiter une bonne heuristique. Ici, il reproduit les choix de BFD pour réinsérer des objets dans une solution partielle.

L'**apprentissage dans LNS** : notre approche est volontairement plus légère et interprétable que les travaux qui remplacent ALNS par un modèle profond. Nous ajoutons deux aides ciblées à des décisions précises.

**Slide actuel : 9 — Architecture globale**

On part d'une solution BFD. À chaque itération ALNS, LinUCB choisit un opérateur de destruction à partir d'un contexte à cinq variables. Après destruction, le modèle GBT note les boîtes faisables pour guider la réinsertion. Le critère d'acceptation est celui du recuit simulé. La meilleure solution rencontrée est conservée.

**Slide actuel : 10 — Initialisation BFD**

BFD trie les objets du plus grand au plus petit, puis insère chaque objet dans la boîte faisable laissant le moins d'espace libre. Si aucune ne convient, une nouvelle boîte est ouverte. Ce point de départ compact donne à ALNS une base solide plutôt qu'une solution initiale trop faible.

---

## 4. Idris Himeur — ALNS, destruction et acceptation

> **Slides couverts : 11 à 18.**

**Slide actuel : 12 — Limite de la recherche locale**

Une recherche locale classique fait de petits changements : c'est rapide, mais cela mène souvent à un optimum local d'où aucune modification marginale ne peut sortir. Dans le bin packing, ces optima sont fréquents car la qualité dépend fortement des groupements d'objets dans les boîtes.

**Slide actuel : 13 — Itération LNS**

ALNS répond à ce problème avec un cycle destruction-réparation. On retire une partie des objets, puis on les réinsère pour reconstruire une solution complète. Modifier plusieurs objets simultanément permet d'explorer un voisinage bien plus large. ALNS y ajoute une sélection adaptative de l'opérateur de destruction.

**Slide actuel : 14 — Opérateurs de destruction**

Trois opérateurs sont disponibles :

- **Aléatoire** : retire des objets uniformément au hasard — favorise l'exploration.
- **Worst-fit** : cible les boîtes les moins remplies — corrige les parties faibles de la solution.
- **Relatif** : retire des objets de tailles proches d'une graine — permet de recombiner des groupes similaires.

Les trois garantissent qu'au moins un objet reste placé.

**Slide actuel : 15 — Rayon de destruction**

On retire entre 5 % et 25 % des objets. En cas de stagnation, la borne supérieure augmente progressivement pour forcer une diversification plus forte.

**Slide actuel : 17 — Acceptation et refroidissement**

Une solution meilleure ou égale est acceptée directement. Une solution moins bonne peut l'être avec une probabilité qui décroît avec la température, selon un refroidissement géométrique. En cas de stagnation, la température est légèrement réchauffée pour éviter un refroidissement prématuré.

**Slide actuel : 18 — Redémarrage de diversification**

En cas de stagnation prolongée, on repart de la meilleure solution connue, on réchauffe la température, on réduit la fenêtre de patience, et on remet le compteur à zéro. Le seul critère d'arrêt dur reste le nombre maximal d'itérations.

---

## 5. Adem Abdelhafidh Diar — Composants d'apprentissage automatique

> **Slides couverts : 19 à 27.**

**Slide actuel : 20 — Deux composants indépendants**

Le premier composant choisit l'opérateur de destruction à chaque itération. Le second choisit la boîte de réinsertion pour chaque objet déplacé. Ils sont indépendants : on peut tester chacun séparément, les deux ensemble, ou aucun.

**Slide actuel : 21 — Phase 1 : démarrage Thompson Sampling**

Pendant les 300 premiers appels, on utilise un Thompson Sampling bêta-bernoulli. Chaque opérateur démarre avec une loi Beta uniforme ; on tire un échantillon pour chacun, on prend le meilleur, et on met à jour uniquement celui choisi. Cette phase explore les opérateurs sans dépendre d'un vecteur de contexte insuffisamment estimé, ce qui atténue le problème de démarrage à froid de LinUCB.

**Slide actuel : 22 — Phase 2 : LinUCB**

Passé les 300 appels, LinUCB prend le relais. Pour chaque opérateur, il calcule un score combinant performance estimée et marge d'exploration (alpha = 0,3). L'opérateur avec le meilleur score est choisi, puis ses paramètres sont mis à jour avec la récompense observée.

**Slide actuel : 23 — Vecteur de contexte**

Le contexte contient cinq variables normalisées entre 0 et 1 : température relative, progression de la stagnation, rapport LB1/coût courant, rayon de destruction relatif, et avancement dans le budget d'itérations.

**Slide actuel : 24 — Récompense**

Si on réduit le nombre de boîtes, la récompense est proportionnelle au gain normalisé par l'écart courant à LB1, bornée à 1. Si la solution est acceptée sans réduction, la récompense vaut 0,2. Si elle est refusée, 0. Cette normalisation valorise davantage les gains obtenus quand on est déjà proche de LB1.

**Slide actuel : 25 — Réparation apprise**

Après une destruction, chaque objet retiré est réinséré du plus grand au plus petit. Pour chaque objet, le modèle note toutes les boîtes faisables et retient la meilleure. S'il n'y en a aucune, une nouvelle boîte est ouverte.

**Slide actuel : 26 — Représentation des caractéristiques**

Le modèle utilise 11 caractéristiques normalisées décrivant l'objet et la boîte candidate : taille normalisée au carré $(s_i/C)^2$, taille normalisée, rang parmi les restants, fraction d'objets restant à réinsérer, charge et espace résiduel de la boîte, espace après insertion, nombre d'objets présents, plus grand et plus petit objet déjà placé, et ratio de remplissage résiduel.

**Slide actuel : 27 — Architecture et données d'entraînement**

Le modèle est un `GradientBoostingClassifier` de scikit-learn utilisé en classeur de paires objet-boîte : les probabilités servent à classer les boîtes faisables. L'entraînement imite BFD — la boîte choisie par BFD reçoit l'étiquette positive, les autres négative. Pour réduire le décalage entraînement/inférence, on utilise deux types de traces : des traces BFD complètes, et des traces issues de solutions BFD partiellement détruites.

---

## 6. Mohamed El Amine Kherroubi — Protocole expérimental et jeux de données

> **Slides couverts : 28 à 30.**

**Slide actuel : 29 — Protocole expérimental**

La graine aléatoire est fixée à 42. ALNS tourne sur 300 itérations, avec une température initiale de $1/\ln 2$ et un coefficient de refroidissement de 0,9995. Pour LinUCB, alpha = 0,3 avec 300 appels de démarrage. Un seul modèle GBT est pré-entraîné sur tous les jeux de données.

La métrique principale est l'écart à LB1 : nombre de boîtes utilisées moins LB1. Un écart nul signifie optimalité certifiée par cette borne.

**Slide actuel : 30 — Jeux de données**

Cinq familles de référence : **Scholl-2** (tailles uniformes), **Falkenauer-T** (structure en triplets, LB1 moins fiable), **Falkenauer-U** (tailles uniformes, autre capacité), **Wäscher** (découpe industrielle), **Hard28** (instances volontairement difficiles). Cette diversité permet de vérifier la robustesse de la méthode au-delà d'un seul type d'instances.

---

## 7. Rayan Boukakiou — Résultats : ablation et généralisation

> **Slides couverts : 31 à 32.**

**Slide actuel : 31 — Test 1 : ablation**

Sur Scholl-2 avec cinq instances de 50 objets, la version sans apprentissage obtient un écart moyen de 0,20. LinUCB seul le ramène à 0,00 avec un temps de calcul faible. Le modèle GBT seul maintient l'écart à 0,20 mais augmente le temps. Les deux composants ensemble donnent aussi 0,20.

La conclusion est claire : sur ces instances, le choix adaptatif des opérateurs est le composant décisif. Le GBT est bien intégré mais son coût n'est pas encore compensé par un gain sur les petites instances.

**Slide actuel : 32 — Test 2 : généralisation multi-jeux**

Sur quatre familles sur cinq — Scholl-2, Falkenauer-U, Wäscher, Hard28 — la méthode reste strictement sous une boîte d'écart. Sur Falkenauer-T, l'écart est exactement 1,00 ; cette famille est isolée car sa structure en triplets affaiblit LB1, ce qui ne reflète pas nécessairement une mauvaise recherche. Globalement, la méthode se généralise sans réglage spécifique par famille.

---

## 8. Idris Yassine Ziadi — Résultats comparatifs

> **Slide couvert : 33.**

**Slide actuel : 33 — Test 3 : comparaison**

Sur Scholl-2 (cinq instances, 50 objets), FFD/BFD, recuit simulé et recherche tabou ont tous un écart de 1,80. L'algorithme génétique descend à 1,00 mais prend plus de temps. Notre ALNS dual-learning atteint 0,20 en environ 0,36 s. ACO obtient 0,00 mais prend en moyenne 3,03 s.

Notre méthode n'est pas toujours la meilleure en qualité pure, mais elle offre le meilleur compromis qualité/temps parmi les alternatives testées. ACO est plus précise mais environ huit fois plus lente.

À noter : les méthodes stochastiques sont évaluées sur une seule exécution. Avec plusieurs graines, les classements pourraient évoluer.

---

## 9. Idris Himeur — Synthèse, limites et conclusion

> **Slides couverts : 34 à 36.**

**Slide actuel : 35 — Synthèse des résultats**

Quatre points ressortent :

1. LinUCB est le composant d'apprentissage le plus impactant : seul, il ferme l'écart de 0,20 à 0,00 dans l'ablation.
2. Le modèle GBT fonctionne mais n'apporte pas encore de gain visible sur les petites instances, où il ajoute surtout du temps.
3. La méthode combinée se généralise bien : quatre familles sur cinq sous une boîte d'écart, Falkenauer-T isolée à 1,00 pour des raisons liées à LB1.
4. Par rapport aux métaheuristiques classiques testées, elle offre un bon compromis qualité/vitesse.

Limites principales : échantillons de cinq instances, une seule exécution par méthode stochastique, borne LB1 simple.

**Slide actuel : 36 — Conclusions**

Notre contribution principale est une ALNS hybride avec deux composants activables séparément : LinUCB pour la sélection des opérateurs de destruction, et un GBT supervisé pour guider la réparation.

Les contributions secondaires sont la chaîne d'apprentissage de la réparation (données augmentées, contrat de 11 caractéristiques, contrôles de qualité, versionnement pour éviter les incompatibilités silencieuses) et le bandit LinUCB avec phase de démarrage Thompson Sampling et récompense normalisée par l'écart à LB1.

Les suites naturelles : tester sur plus d'instances, moyenner sur plusieurs graines, adopter une borne plus serrée comme Martello-Toth L2, et explorer une réparation par renforcement de bout en bout.

---

## 10. Adem Abdelhafidh Diar — Mot de fin

> **Slide couvert : 36, après la conclusion technique.**

**Slide actuel : 36 — Mot de fin**

Notre objectif n'était pas de remplacer les heuristiques classiques par un modèle opaque. Nous avons gardé ALNS et ajouté de l'apprentissage uniquement aux décisions répétées où il peut aider. Les résultats montrent que cette approche est prometteuse, surtout pour le choix adaptatif des opérateurs.

Merci pour votre attention. Nous sommes prêts pour vos questions.
