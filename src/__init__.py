"""
Module principal du projet ML de détection de fatigue
"""

__version__ = "1.0.0"
__author__ = "aziz-hadjayed"

# Importer les fonctions clés
from .config import DATA_PROCESSED, DATA_RAW, MODEL_PARAMS
from .data_prep import data_clean_1, clean_data_2, encode_features
from .models import get_random_forest_model, get_svm_model
from .train import evaluate_model, train_model

__all__ = [
    "DATA_RAW",
    "DATA_PROCESSED",
    "MODEL_PARAMS",
    "data_clean_1",
    "clean_data_2",
    "encode_features",
    "get_random_forest_model",
    "get_svm_model",
    "train_model",
    "evaluate_model",
]
