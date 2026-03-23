# ------------------------------------------// Importations
import json

from src.config import *
from src.data_prep import *
from src.models import *
from src.train import *

# ------------------------------------------// body
if __name__ == "__main__":
    load_dotenv()
    # 1. Charger et préparer les données
    df = clean_data(df)
    X, y = encode_features(df)
    X = normalize_features(X)








 













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





