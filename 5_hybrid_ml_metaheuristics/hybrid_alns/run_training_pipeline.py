#!/usr/bin/env python3
"""
Script d'orchestration automatique du workflow d'entraînement ML.

Usage:
    python run_training_pipeline.py --all              (pipeline complet)
    python run_training_pipeline.py --stage 1          (juste étape 1)
    python run_training_pipeline.py --quick            (v1 sans plots longs)
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime


def run_command(cmd: list[str], description: str) -> int:
    """Exécute une commande shell et track le résultat."""
    print("\n" + "=" * 80)
    print(f"📌 {description}")
    print("=" * 80)
    print(f"Commande: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ {description} — SUCCÈS\n")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} — ERREUR (code {e.returncode})\n")
        return e.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrer automatiquement le pipeline d'entraînement ML"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Lancer le pipeline COMPLET (v1 → DAgger → v2)",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Lancer juste UNE étape spécifique (1-5)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Version rapide : v1 seulement, sans learning curves",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Nombre de workers parallèles (défaut: 4)",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=5000,
        help="Nombre d'instances synthétiques (défaut: 5000)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Sauter la génération de graphiques",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Nombre de folds cross-validation (défaut: 5)",
    )

    args = parser.parse_args()

    # Logging
    log_file = Path("training_log.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write_text(f"Pipeline démarré : {timestamp}\n\n")

    errors = []

    # ========================================================================
    # ÉTAPE 1 : Training v1 (synthétique pur)
    # ========================================================================
    if args.all or args.stage == 1 or args.quick:
        cmd = [
            "python",
            "train_repair_model_improved.py",
            f"--instances={args.instances}",
            "--n-min=50",
            "--n-max=200",
            "--max-negatives=5",
            "--seed=0",
            f"--workers={args.workers}",
            "--output=repair_model_v1.pkl",
            f"--cv-folds={args.cv_folds}",
        ]

        if args.quick or args.no_plots:
            cmd.append("--no-learning-curves")

        err = run_command(
            cmd,
            "ÉTAPE 1 : Entraînement modèle v1 (synthétique pur, 5K instances)",
        )
        if err:
            errors.append("Étape 1 échouée")
            if args.stage == 1:
                return 1

    if args.quick:
        print("\n✅ Mode QUICK terminé. Modèle v1 prêt : repair_model_v1.pkl")
        return 0

    # ========================================================================
    # ÉTAPE 2 : Collecte DAgger-lite
    # ========================================================================
    if args.all or args.stage == 2:
        cmd = [
            "python",
            "collect_alns_states.py",
            "--model-path=repair_model_v1.pkl",
            "--instances=500",
            "--n-min=50",
            "--n-max=200",
            "--iterations=200",
            "--max-negatives=5",
            "--seed=1",
            "--output=alns_states_v1.pkl",
        ]

        err = run_command(
            cmd,
            "ÉTAPE 2 : Collecte ALNS states pour DAgger-lite (500 instances)",
        )
        if err:
            errors.append("Étape 2 échouée")
            if args.stage == 2:
                return 1

    # ========================================================================
    # ÉTAPE 3 : Fine-tuning v2 (synthétique + ALNS)
    # ========================================================================
    if args.all or args.stage == 3:
        cmd = [
            "python",
            "train_repair_model_improved.py",
            f"--instances={args.instances}",
            "--n-min=50",
            "--n-max=200",
            "--max-negatives=5",
            "--seed=0",
            f"--workers={args.workers}",
            "--augment-with=alns_states_v1.pkl",
            "--output=repair_model_v2.pkl",
            f"--cv-folds={args.cv_folds}",
        ]

        if args.no_plots:
            cmd.append("--no-plots")

        err = run_command(
            cmd,
            "ÉTAPE 3 : Fine-tuning modèle v2 (synthétique + 500 ALNS states)",
        )
        if err:
            errors.append("Étape 3 échouée")
            if args.stage == 3:
                return 1

    # ========================================================================
    # ÉTAPE 4 : Validation sur benchmark
    # ========================================================================
    if args.all or args.stage == 4:
        print("\n" + "=" * 80)
        print("📌 ÉTAPE 4 : Validation sur dataset benchmark")
        print("=" * 80)
        print(
            """
Pour tester le modèle v2 sur données réelles Falkenauer, lancez:

    python benchmark.py \\
        --solver 5_hybrid_ml_metaheuristics/hybrid_alns/solver.py \\
        --dataset falkenauer-t \\
        --method-args "model_path=repair_model_v2.pkl,max_iterations=5000"

Alternativement, sur Falkenauer U (plus gros) :

    python benchmark.py \\
        --solver 5_hybrid_ml_metaheuristics/hybrid_alns/solver.py \\
        --dataset falkenauer-u \\
        --method-args "model_path=repair_model_v2.pkl,max_iterations=5000"
"""
        )

    # ========================================================================
    # RAPPORT FINAL
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 RAPPORT FINAL")
    print("=" * 80)

    if errors:
        print(f"\n❌ {len(errors)} erreur(s) détectée(s) :")
        for err in errors:
            print(f"   - {err}")
        return 1
    else:
        print("\n✅ Pipeline COMPLET terminé avec succès!\n")
        print("Modèles générés :")
        print("  • repair_model_v1.pkl  (baseline synthétique)")
        print("  • repair_model_v2.pkl  (production avec DAgger-lite)")
        print("\nGéométrie/visualisations :")
        print("  • learning_curves.png")
        print("  • roc_curve.png")
        print("  • feature_importance.png")
        print("\nProchaine étape : Tester avec benchmark.py")
        return 0


if __name__ == "__main__":
    sys.exit(main())
