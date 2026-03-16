# ------------------------------------------// Importations
import os

# ------------------------------------------// Chemins
DATA_RAW = os.path.join("data", "02_interim/dataset.csv")
DATA_PROCESSED = os.path.join("data", "03_processed")


# ------------------------------------------// Hyperparamètres
MODEL_PARAMS = {
    "RandomForest": {"n_estimators": 100, "max_depth": 10},
    "SVM": {"kernel": "rbf", "C": 1.0},
}
