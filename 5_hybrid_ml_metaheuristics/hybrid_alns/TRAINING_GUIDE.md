"""
GUIDE COMPLET : PROCESSUS D'ENTRAÎNEMENT OFFLINE DU MODÈLE ML

Responsable : [Votre nom] - Partie ML Logistic Regression (GradientBoosting)
Date : Mai 2026
"""

================================================================================
SECTION 1 : PRÉALABLES ET VÉRIFICATIONS
================================================================================

1.1. Dépendances requises
    ✅ numpy          ≥ 1.20   (calculs matriciels)
    ✅ scikit-learn   ≥ 1.0    (ML models, preprocessing, metrics)
    ✅ numba          (OPTIONNEL, accélère 100x les features)
    ✅ tqdm           (OPTIONNEL, barres de progression)
    ✅ matplotlib     (OPTIONNEL, visualisations)

Installation :
    pip install numpy scikit-learn numba tqdm matplotlib

1.2. Fichiers requis
    ✓ features.py                    (définition des 11 features)
    ✓ train_repair_model.py          (générateur de données)
    ✓ train_repair_model_improved.py (entraînement amélioré - NOUVEAU)
    ✓ collect_alns_states.py         (génération DAgger-lite)
    ✓ solver.py                      (validation finale)

1.3. Données disponibles
    Synthétiques   : Générées on-the-fly (illimitées) ✅
    Réelles        : datasets/Falkenauer/{t60,u120,u1000}, datasets/Scholl/
                     (À intégrer dans pipeline futur)

================================================================================
SECTION 2 : FLUX DE TRAVAIL RECOMMANDÉ (WORKFLOW)
================================================================================

ÉTAPE 1 : Entraînement INITIAL sur synthétique pur
┌─────────────────────────────────────────────────────────────┐
│ python train_repair_model_improved.py \                      │
│   --instances 5000 \                                         │
│   --n-min 50 --n-max 200 \                                   │
│   --max-negatives 5 \                                        │
│   --seed 0 \                                                 │
│   --workers 4 \                                              │
│   --output repair_model_v1.pkl \                             │
│   --cv-folds 5 \                                             │
│   --no-learning-curves          (passe rapide)               │
└─────────────────────────────────────────────────────────────┘

Sortie attendue :
  ✅ repair_model_v1.pkl       (modèle entraîné, ~5 MB)
  ✅ learning_curves.png        (graphiques overfitting)
  ✅ roc_curve.png              (métrique ROC-AUC)
  ✅ feature_importance.png     (quelles features comptent)
  ✅ Rapport console             (métriques détaillées)

Durée : ~2-5 minutes (selon CPU)

================================================================================

ÉTAPE 2 : Réduction du COVARIATE SHIFT (DAgger-lite)
┌─────────────────────────────────────────────────────────────┐
│ python collect_alns_states.py \                              │
│   --model-path repair_model_v1.pkl \                         │
│   --instances 500 \                                          │
│   --n-min 50 --n-max 200 \                                   │
│   --iterations 200 \                                         │
│   --max-negatives 5 \                                        │
│   --seed 1 \                                                 │
│   --output alns_states_v1.pkl                                │
└─────────────────────────────────────────────────────────────┘

Que se passe-t-il :
  1. Lance le modèle v1 sur 500 instances
  2. Capture les VRAIS états rencontrés (pas du BFD)
  3. Étiquète avec oracle BFD
  4. Sauvegarde en pickle → ~10K-50K exemples réels

Sortie : alns_states_v1.pkl (~1-5 MB)
Durée : ~5-10 minutes

================================================================================

ÉTAPE 3 : Fine-tuning avec données augmentées
┌─────────────────────────────────────────────────────────────┐
│ python train_repair_model_improved.py \                      │
│   --instances 5000 \                                         │
│   --n-min 50 --n-max 200 \                                   │
│   --max-negatives 5 \                                        │
│   --seed 0 \                                                 │
│   --workers 4 \                                              │
│   --augment-with alns_states_v1.pkl \                        │
│   --output repair_model_v2.pkl \                             │
│   --cv-folds 5                                               │
└─────────────────────────────────────────────────────────────┘

Que change :
  - Dataset de 5K instances + 500 ALNS states
  - Réduit covariate shift (le modèle voit états ALNS réels)
  - Meilleure généralisation attendue

Sortie : repair_model_v2.pkl (meilleur modèle)
Durée : ~3-6 minutes

================================================================================

ÉTAPE 4 (Optionnel) : Hyperparameter Tuning fin
┌─────────────────────────────────────────────────────────────┐
│ python train_repair_model_improved.py \                      │
│   --instances 5000 \                                         │
│   --augment-with alns_states_v1.pkl \                        │
│   --grid-search \                                            │
│   --output repair_model_v2_tuned.pkl                         │
└─────────────────────────────────────────────────────────────┘

⚠️  ATTENTION : Très long (1-2 heures!)
Testez les 3 hyperparams : n_estimators, max_depth, learning_rate
À faire une seule fois après avoir validé la pipeline.

Durée : ~60-120 minutes

================================================================================

ÉTAPE 5 : Validation finale
┌─────────────────────────────────────────────────────────────┐
│ python benchmark.py \                                        │
│   --solver 5_hybrid_ml_metaheuristics/hybrid_alns/solver.py \│
│   --dataset falkenauer-t \                                   │
│   --method-args "model_path=repair_model_v2.pkl"             │
└─────────────────────────────────────────────────────────────┘

Vérifiez :
  ✓ Aucune erreur de chargement du modèle
  ✓ ALNS tourne sans crash
  ✓ Solution finale est meilleure qu'approche naive

================================================================================
SECTION 3 : EXPLICATIONS DÉTAILLÉES
================================================================================

3.1. Pourquoi le workflow en 5 étapes ?

    Étape 1 (v1) : Baseline rapide
    ├─ Valide que la pipeline fonctionne
    ├─ Fournit métriques de base
    └─ Teste sur instances synthétiques pures

    Étape 2 (DAgger) : Collecte d'expériences réelles
    ├─ Exécute ALNS dans ses conditions réelles
    ├─ Capture états que BFD JAMAIS produit
    └─ Étiquette avec oracle → données de qualité

    Étape 3 (v2) : Fine-tuning avec mélange
    ├─ Combine synthétique (5K) + réel (500)
    ├─ Réduit gap train/test
    └─ Modèle final de production

    Étape 4 (Tuning) : Optimisation sans regrets
    ├─ Recherche exhaustive des meilleurs hyperparams
    ├─ À faire UNE FOIS après validation v2
    └─ Gagne ~2-5% de performance

    Étape 5 (Validation) : Test en conditions réelles
    ├─ Vérifiez que ALNS se comporte bien
    ├─ Comparez vs autres méthodes
    └─ Validez déploiement


3.2. Paramètres clés expliqués

    --instances 5000
        Nombre d'instances synthétiques générées.
        ✓ 5000 = bon par défaut
        ↑ 10000 = meilleure généralisation (plus long)
        ↓ 1000  = rapide mais moins robuste

    --n-min 50, --n-max 200
        Plage de taille des instances (nombre d'articles).
        ✓ [50-200] = bon par défaut
        ↑ [100-500] = plus gros problèmes
        ↓ [20-100] = plus petits problèmes

    --max-negatives 5
        Limite les exemples négatifs par exemple positif.
        ✓ 5 = équilibre train/test bien
        ↑ 10 = plus d'équilibre (plus de données)
        ↓ 2-3 = moins d'équilibre mais apprentissage plus rapide

    --workers 4
        Nombre de processus parallèles pour génération données.
        ✓ CPU count / 2 (ex: 8 cores → 4 workers)
        ⚠️  Trop haut = surcharge mémoire
        ✓ Mettez 1 si pb de mémoire

    --seed 0
        Graine RNG pour reproductibilité.
        ✓ Même seed = même données générées
        ✓ À utiliser pour comparaisons

    --cv-folds 5
        Cross-validation k-fold.
        ✓ 5 = standard
        ↑ 10 = plus robuste (plus lent)

    --grid-search
        Active recherche exhaustive hyperparams.
        ⚠️  Très long, fait SEULEMENT après avoir validé v2


3.3. Comment interpréter les métriques ?

    ROC-AUC (PRINCIPALE) : 0-1
    ├─ > 0.85 : excellent! Le modèle rank bien les bacs
    ├─ 0.75-0.85 : bon, acceptable
    ├─ 0.65-0.75 : moyen, améliorer
    └─ < 0.65 : mauvais, revoir features/données

    F1-Score : 0-1 (équilibre precision/recall)
    ├─ Moyenne harmonique de Precision et Recall
    └─ À optimiser pour minimiser faux positifs ET faux négatifs

    Precision : 0-1 (vrais positifs / prédits positifs)
    ├─ Si precision=0.9 : quand le modèle dit OUI, 90% correct
    ├─ Haut precision = fiable, peu faux positifs
    └─ Important pour ALNS (on veut bon choix, pas devinette)

    Recall : 0-1 (vrais positifs / réels positifs)
    ├─ Si recall=0.8 : capture 80% des bacs optimaux
    ├─ Haut recall = ne loupe pas les bons choix
    └─ Moins critique que Precision en ALNS

    Confusion Matrix :
    ├─ True Positives (TP)  : bac optimal prédit optimal ✓✓
    ├─ False Positives (FP) : bac pas optimal prédit optimal ✗✓
    ├─ False Negatives (FN) : bac optimal prédit pas optimal ✓✗
    └─ True Negatives (TN)  : bac pas optimal prédit pas optimal ✗✗


3.4. Covariate Shift et pourquoi DAgger-lite ?

    PROBLÈME :
        Model trained on    : états BFD (bacs bien chargés)
        Model used on       : états ALNS (bacs partiels, aléatoires)
        → Distribution très différente! Performance dégrade.

    SOLUTION DAgger-lite :
        1. Exécutez ALNS avec model v1
        2. Capturez les VRAIS états rencontrés
        3. Étiquetez avec oracle BFD
        4. Réentraînez v2 avec mélange
        → Model v2 "voit" pendant entraînement états qu'il verra en test

    RÉSULTAT :
        ✓ Meilleure adaptation aux états ALNS réels
        ✓ Généralement +5-15% de performance
        ✓ Coûte juste temps collecte données (~5-10 min)


3.5. Feature Importance : quelles features comptent ?

    Après entraînement, matplotlib affiche top 5 features.
    Exemple :
        1. remaining          (fraction articles à placer)
        2. slack_after        (résidu après placement)
        3. bin_load           (charge actuell bac)
        4. item_size          (taille article)
        5. size_rank          (rang taille article)

    Interprétation :
        ✓ remaining + slack_after dominent → bac dynamique
        ✓ item_size + size_rank importants → article importe
        ✓ bin_load pertinent → charge est signal
        ✗ Si feature_importance = plat → features non-informatives


================================================================================
SECTION 4 : COMMANDES RAPIDES (ONE-LINERS)
================================================================================

Quick baseline (30 sec) :
    python train_repair_model_improved.py --instances 500 --no-plots

Quick full (5 min) :
    python train_repair_model_improved.py --instances 5000 --workers 4

With DAgger augmentation (10 min) :
    python collect_alns_states.py --model-path repair_model_v1.pkl --instances 500
    python train_repair_model_improved.py --instances 5000 --augment-with alns_states_v1.pkl

Full pipeline (20 min) :
    python train_repair_model_improved.py --instances 5000 --workers 4 --output v1.pkl
    python collect_alns_states.py --model-path v1.pkl --instances 500 --output states.pkl
    python train_repair_model_improved.py --instances 5000 --augment-with states.pkl --output v2.pkl


================================================================================
SECTION 5 : TROUBLESHOOTING
================================================================================

❌ "Error: numba not found" 
    → OK, utilise fallback Python pur (plus lent mais OK)
    → Pour accélérer : pip install numba

❌ "ROC-AUC < 0.70"
    → Augmentez --instances à 10000 ou 20000
    → Vérifiez que features.py match train_repair_model.py
    → Inspectez feature_importance.png

❌ "MemoryError during data generation"
    → Réduisez --workers (ex: 1 au lieu de 4)
    → Réduisez --instances
    → Augmentez RAM disponible

❌ "Model file corrupted" après sauvegarde
    → Vérifiez espace disque disponible
    → Vérifiez permissions d'écriture
    → Relancez depuis début

❌ "Cross-validation too slow"
    → Réduisez --cv-folds (ex: 3 au lieu de 5)
    → Augmentez --workers pour parallélisation
    → Sautez cross-validation en prod (-skip-cv)


================================================================================
SECTION 6 : PROCHAINES ÉTAPES RECOMMANDÉES (POUR VOUS)
================================================================================

À court terme (cette semaine) :
    1. ✅ Lancez train_repair_model_improved.py avec defaults
    2. ✅ Vérifiez ROC-AUC > 0.75
    3. ✅ Regardez feature_importance.png
    4. ✅ Testez modèle dans benchmark.py

À moyen terme (semaine 2) :
    1. Lancez DAgger-lite (collect_alns_states.py)
    2. Entraînez v2 avec augmentation
    3. Comparez v1 vs v2 performance

À plus long terme (semaine 3+) :
    1. Intégrez données Falkenauer réelles (script à écrire)
    2. Grid search pour tuning final
    3. Analyse complète des résultats

================================================================================
FICHIERS GÉNÉRÉS APRÈS ENTRAÎNEMENT
================================================================================

repair_model_v1.pkl          (modèle final, ~5 MB)
├─ .model                    (GradientBoostingClassifier entraîné)
├─ .scaler                   (StandardScaler fitted)
├─ .feature_version          (validation version)
├─ .metrics                  (ROC-AUC, F1, Precision, Recall)
└─ .cv_scores                (cross-validation scores)

learning_curves.png          (graphique overfitting)
roc_curve.png                (courbe ROC avec AUC)
feature_importance.png       (barres importance features)

================================================================================
"""
