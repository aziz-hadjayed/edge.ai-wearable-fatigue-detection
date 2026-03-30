"""
Module principal du projet ML de détection de fatigue
"""

__version__ = "1.0.0"
__author__ = "aziz-hadjayed"

from .config import DATA_PROCESSED, DATA_RAW, MODEL_PARAMS
from .data_prep import data_clean_1, clean_data_2, encode_features

__all__ = [
    "DATA_RAW",
    "DATA_PROCESSED",
    "MODEL_PARAMS",
    "data_clean_1",
    "clean_data_2",
    "encode_features",
]

