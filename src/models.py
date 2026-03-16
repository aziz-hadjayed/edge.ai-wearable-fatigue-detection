# Exemple
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def get_random_forest_model(params):
    return RandomForestClassifier(**params)


def get_svm_model(params):
    return SVC(**params)


models = {"RandomForest": get_random_forest_model, "SVM": get_svm_model}
