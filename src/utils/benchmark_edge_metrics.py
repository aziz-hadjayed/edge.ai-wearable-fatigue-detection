import subprocess
import json
from pathlib import Path
import sys
import pandas as pd

# Ajouter le root et src au path pour les imports (meme logique que apply_smote.py)
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from config import MODELS_DIR, BASE_DIR


def find_model_file(model_dir: Path):
    """
    Cherche un fichier modele exploitable par stedgeai dans model_dir.
    Priorite : *_int8.tflite, sinon *.onnx (fallback).
    Retourne (chemin, format) ou (None, None) si rien trouve.
    """
    tflite_candidates = sorted(model_dir.glob("*_int8.tflite"))
    if tflite_candidates:
        return tflite_candidates[0], "tflite"

    onnx_candidates = sorted(model_dir.glob("*.onnx"))
    if onnx_candidates:
        return onnx_candidates[0], "onnx"

    return None, None


def run_stedgeai_analyze(model_path: Path, output_dir: Path, target="stm32h7"):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "stedgeai", "analyze",
        "--target", target,
        "--model", str(model_path),
        "--output", str(output_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERREUR] {model_path.name}: {result.stderr}")
        return None
    # stdout peut aussi contenir des infos utiles a parser
    print(result.stdout)
    return output_dir


def parse_report(output_dir: Path):
    report_files = list(output_dir.glob("*.json"))
    if not report_files:
        print(f"  [WARN] Pas de rapport JSON trouve dans {output_dir}")
        return {}

    with open(report_files[0]) as f:
        data = json.load(f)

    mem = data.get("memory_footprint", {})
    weights        = mem.get("weights", 0)
    activations    = mem.get("activations", 0)
    kernel_flash   = mem.get("kernel_flash", 0)
    kernel_ram     = mem.get("kernel_ram", 0)
    toolchain_flash = mem.get("toolchain_flash", 0)
    toolchain_ram   = mem.get("toolchain_ram", 0)

    flash_bytes = weights + kernel_flash + toolchain_flash
    ram_bytes   = activations + kernel_ram + toolchain_ram

    # MACC total = somme des macc de chaque noeud/couche du graphe
    total_macc = 0
    graphs = data.get("graphs", [])
    if graphs:
        total_macc = sum(node.get("macc", 0) for node in graphs[0].get("nodes", []))

    # Infos complementaires sur le modele source
    input_model = {}
    tools = data.get("environment", {}).get("tools", [])
    if tools:
        input_model = tools[0].get("input_model", {})

    return {
        "flash_kb": round(flash_bytes / 1024, 2),
        "ram_kb": round(ram_bytes / 1024, 2),
        "weights_kb": round(weights / 1024, 2),
        "activations_kb": round(activations / 1024, 2),
        "macc": total_macc,
        "n_params": input_model.get("n_params"),
        "model_file_size_kb": round(input_model.get("size", 0) / 1024, 2),
    }


def analyze_all_models(models_dir: Path, reports_dir: Path):
    rows = []

    # On parcourt chaque sous-dossier de modele (CNN, ESN, LGBM, QRC, ...)
    model_dirs = sorted(d for d in models_dir.iterdir() if d.is_dir())

    for model_dir in model_dirs:
        model_name = model_dir.name

        model_path, model_format = find_model_file(model_dir)
        if model_path is None:
            print(f"[SKIP] {model_name}: aucun fichier .tflite ou .onnx trouve")
            continue

        print(f"Analyse : {model_name} ({model_format} -> {model_path.name})")

        output_dir = reports_dir / model_name
        result_dir = run_stedgeai_analyze(model_path, output_dir)
        if result_dir is None:
            continue

        metrics = parse_report(result_dir)
        metrics["model"] = model_name
        metrics["source_format"] = model_format
        metrics["source_file"] = model_path.name
        rows.append(metrics)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    reports_dir = BASE_DIR / "edge_reports"
    df_edge = analyze_all_models(MODELS_DIR, reports_dir)
    df_edge.to_csv(BASE_DIR / "edge_metrics.csv", index=False)
    print(df_edge)