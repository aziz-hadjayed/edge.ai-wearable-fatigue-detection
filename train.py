import os
import sys
import subprocess
import time
from pathlib import Path
from glob import glob

def train_all_models():
    """
    Découvre et exécute séquentiellement tous les scripts de training
    dans src/models/train_*.py.
    """
    # Répertoire racine du projet (là où se trouve ce script)
    root_dir = Path(__file__).resolve().parent
    models_dir = root_dir / "src" / "models"
    
    # Trouver tous les scripts train_*.py
    scripts = sorted(glob(str(models_dir / "train_*.py")))
    
    if not scripts:
        print("❌ Aucun script de training trouvé dans src/models/")
        return

    print("=" * 60)
    print(f"🚀 Démarrage du training global ({len(scripts)} scripts détectés)")
    print("=" * 60)

    results = []

    for script_path in scripts:
        script_name = os.path.basename(script_path)
        
        # Ignorer les fichiers vides (comme train_tcn.py)
        if os.path.getsize(script_path) == 0:
            print(f"⚠️  Skip {script_name} (fichier vide)")
            continue

        print(f"\n▶️  ENTRAÎNEMENT : {script_name}")
        print("-" * 40)
        
        start_time = time.time()
        try:
            # Exécuter le script en tant que processus séparé
            # On utilise sys.executable pour s'assurer d'utiliser le même interpréteur Python
            # Le cwd est mis à la racine pour la cohérence
            process = subprocess.run(
                [sys.executable, script_path],
                cwd=root_dir,
                check=True
            )
            elapsed = time.time() - start_time
            print(f"\n✅ Terminé : {script_name} ({elapsed/60:.1f} min)")
            results.append((script_name, "SUCCESS", elapsed))
        except subprocess.CalledProcessError:
            elapsed = time.time() - start_time
            print(f"\n❌ Erreur lors de l'exécution de {script_name}")
            results.append((script_name, "FAILED", elapsed))
        except KeyboardInterrupt:
            print("\n🛑 Interruption par l'utilisateur. Arrêt du training global.")
            break
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ Erreur inattendue pour {script_name} : {e}")
            results.append((script_name, "ERROR", elapsed))

    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DU TRAINING GLOBAL")
    print("=" * 60)
    if not results:
        print("Aucun script n'a été exécuté.")
    else:
        for name, status, duration in results:
            icon = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⚠️"
            print(f"{icon} {name:<25} | {status:<10} | Durée: {duration/60:.1f} min")
    print("=" * 60)

if __name__ == "__main__":
    train_all_models()
import subprocess