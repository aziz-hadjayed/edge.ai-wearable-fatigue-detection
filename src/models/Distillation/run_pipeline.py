"""
Orchestration du pipeline de distillation :
  Teacher (ViT-1D) → Student (CNN micro) → rapport comparatif vs CNN-D1.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import BASE_DIR, DATA_MODEL_READY, METRICS_PATH, REPORTS_DIR

from models.Distillation.Teacher import MODEL_NAME as TEACHER_NAME, main as train_teacher
from models.Distillation.Student import MODEL_NAME as STUDENT_NAME, main as train_student

CNN_BASELINE_NAME = "CNN_1D_WITH_1HZ"
REPORT_PATH = REPORTS_DIR / "distillation_pipeline_report.json"
REPORT_MD_PATH = REPORTS_DIR / "distillation_pipeline_report.md"


def check_dataset():
    if not DATA_MODEL_READY.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATA_MODEL_READY}")
    print(f"✅ Dataset OK : {DATA_MODEL_READY}")


def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    try:
        with open(METRICS_PATH, "r") as f:
            content = f.read().strip()
        return json.loads(content) if content else {}
    except json.JSONDecodeError:
        return {}


def summarize_model(metrics, name):
    if name not in metrics:
        return {"status": "missing", "name": name}
    m = metrics[name]
    return {
        "status": "ok",
        "name": name,
        "mean_f1": m.get("mean_f1"),
        "std_f1": m.get("std_f1"),
        "mean_bal_acc": m.get("mean_bal_acc"),
        "std_bal_acc": m.get("std_bal_acc"),
        "n_folds": len(m.get("folds", [])),
    }


def build_comparison(metrics):
    teacher = summarize_model(metrics, TEACHER_NAME)
    student = summarize_model(metrics, STUDENT_NAME)
    cnn = summarize_model(metrics, CNN_BASELINE_NAME)

    rows = [r for r in (teacher, student, cnn) if r["status"] == "ok"]
    best_f1 = max(rows, key=lambda r: r["mean_f1"] or 0) if rows else None
    best_bal = max(rows, key=lambda r: r["mean_bal_acc"] or 0) if rows else None

    deltas = {}
    if student["status"] == "ok" and cnn["status"] == "ok":
        deltas["student_vs_cnn_f1"] = student["mean_f1"] - cnn["mean_f1"]
        deltas["student_vs_cnn_bal_acc"] = student["mean_bal_acc"] - cnn["mean_bal_acc"]
    if student["status"] == "ok" and teacher["status"] == "ok":
        deltas["student_vs_teacher_f1"] = student["mean_f1"] - teacher["mean_f1"]
        deltas["student_vs_teacher_bal_acc"] = student["mean_bal_acc"] - teacher["mean_bal_acc"]

    return {
        "teacher": teacher,
        "student": student,
        "cnn_baseline": cnn,
        "best_f1_model": best_f1["name"] if best_f1 else None,
        "best_bal_acc_model": best_bal["name"] if best_bal else None,
        "deltas": deltas,
    }


def write_report(comparison, elapsed_s, steps):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now().isoformat(),
        "elapsed_seconds": elapsed_s,
        "steps": steps,
        "dataset": str(DATA_MODEL_READY),
        "comparison": comparison,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    lines = [
        "# Rapport pipeline Distillation",
        "",
        f"- **Généré** : {report['generated_at']}",
        f"- **Durée** : {elapsed_s:.0f} s",
        f"- **Dataset** : `{DATA_MODEL_READY}`",
        "",
        "## Métriques LOSO (moyenne ± écart-type implicite dans JSON)",
        "",
        "| Modèle | F1 macro | Bal. accuracy | Folds |",
        "|--------|----------|---------------|-------|",
    ]
    for key in ("teacher", "student", "cnn_baseline"):
        row = comparison[key]
        if row["status"] == "ok":
            lines.append(
                f"| {row['name']} | {row['mean_f1']:.4f} | {row['mean_bal_acc']:.4f} | {row['n_folds']} |"
            )
        else:
            lines.append(f"| {row['name']} | — | — | — |")

    if comparison.get("deltas"):
        lines += ["", "## Deltas Student", ""]
        for k, v in comparison["deltas"].items():
            lines.append(f"- `{k}` : {v:+.4f}")

    lines += [
        "",
        "## Recommandation",
        "",
        f"- Meilleur F1 macro : **{comparison.get('best_f1_model', 'N/A')}**",
        f"- Meilleure bal. accuracy : **{comparison.get('best_bal_acc_model', 'N/A')}**",
        "",
        "Export STM32 (TFLite INT8 + `.h`) : `models_saved/Distillation/`",
    ]
    REPORT_MD_PATH.write_text("\n".join(lines))
    print(f"\n📄 Rapport JSON : {REPORT_PATH}")
    print(f"📄 Rapport MD   : {REPORT_MD_PATH}")


def run_pipeline(skip_teacher=False, skip_student=False):
    t0 = time.time()
    steps = []

    print("\n" + "=" * 60 + "\nPIPELINE DISTILLATION — Edge AI Fatigue\n" + "=" * 60)
    check_dataset()
    steps.append("dataset_ok")

    if not skip_teacher:
        print("\n>>> Étape 1/2 : Teacher (Transformer ViT-1D)")
        train_teacher()
        steps.append("teacher_done")
    else:
        print("\n>>> Étape 1/2 : Teacher — ignoré (skip_teacher=True)")
        steps.append("teacher_skipped")

    if not skip_student:
        print("\n>>> Étape 2/2 : Student (CNN micro + distillation)")
        train_student()
        steps.append("student_done")
    else:
        print("\n>>> Étape 2/2 : Student — ignoré (skip_student=True)")
        steps.append("student_skipped")

    metrics = load_metrics()
    comparison = build_comparison(metrics)
    elapsed = time.time() - t0
    write_report(comparison, elapsed, steps)

    print("\n" + "=" * 60 + "\nCOMPARAISON FINALE\n" + "=" * 60)
    for label, key in [("Teacher", "teacher"), ("Student", "student"), ("CNN-D1", "cnn_baseline")]:
        row = comparison[key]
        if row["status"] == "ok":
            print(f"  {label:10s}  F1={row['mean_f1']:.4f}  Bal.Acc={row['mean_bal_acc']:.4f}")
        else:
            print(f"  {label:10s}  — métriques absentes (entraîner le modèle d'abord)")

    if comparison.get("deltas"):
        print("\n  Deltas Student:")
        for k, v in comparison["deltas"].items():
            print(f"    {k}: {v:+.4f}")

    print(f"\n✅ Pipeline terminé en {elapsed:.0f} s")
    return comparison


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pipeline distillation Teacher → Student")
    parser.add_argument("--skip-teacher", action="store_true", help="Ne pas ré-entraîner le Teacher")
    parser.add_argument("--skip-student", action="store_true", help="Ne pas entraîner le Student")
    args = parser.parse_args()
    run_pipeline(skip_teacher=args.skip_teacher, skip_student=args.skip_student)


if __name__ == "__main__":
    main()
