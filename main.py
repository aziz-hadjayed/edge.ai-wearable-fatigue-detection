# ------------------------------------------// Importations
import json

from src.config import *
from src.data_prep import *
from src.train import *

# ------------------------------------------// body
if __name__ == "__main__":
    print("=" * 60)
    print("CHARGEMENT DU DATASET")
    print("=" * 60)
    # df = pd.read_csv(DATA_RAW)
    print(f"✔ Répertoire source : {DATA_RAW}")
 
    # ── Pipeline en 3 étapes ──────────────────────────────────────────────
    df = data_clean_1(DATA_RAW)
    visualize_data(df)
    df = clean_data_2(df)
    X, y = encode_features(df)
    X, scaler = normalize_features(X)

    # ── Sauvegarde ────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
 
    df_meta  = df[[COL_PARTICIPANT, COL_SESSION, COL_TIMESTAMP]].reset_index(drop=True)
    df_final = pd.concat([df_meta, X, y.rename(COL_LABEL)], axis=1)
    df_final.to_csv(OUTPUT_PATH, index=False)
    # ── Résumé ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"✔ Fichier sauvegardé : {OUTPUT_PATH}")
    print(f"✔ Shape finale       : {df_final.shape}")
    print(f"✔ NaN restants       : {df_final.isna().sum().sum()}")
    print(f"✔ RAM X (float32)    : {X.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"✔ Distribution label :")
    print(y.value_counts().sort_index().to_string())
 
    print("\n" + "=" * 60)
    print("⚠ RAPPEL LOSO — éviter le data leakage")
    print("=" * 60)
    print("""
  for test_subject in participants:
      df_train = df[df.participant != test_subject]
      df_test  = df[df.participant == test_subject]
 
      df_train         = clean_data(df_train)
      X_train, y_train = encode_features(df_train)
      X_train, scaler  = normalize_features(X_train, fit=True)
 
      df_test          = clean_data(df_test)
      X_test,  y_test  = encode_features(df_test)
      X_test,  _       = normalize_features(X_test, scaler=scaler, fit=False)
    """)



 













    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 3. Entraîner plusieurs modèles
    results = {}
    for model_name, model_fn in models.items():
        model = model_fn(MODEL_PARAMS[model_name])
        train_model(model, X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[model_name] = metrics

        # Sauvegarder
        joblib.dump(model, f"{MODELS_DIR}/{model_name}.joblib")

    # 4. Sauvegarder les résultats
    with open("reports/metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Pipeline terminé!")





