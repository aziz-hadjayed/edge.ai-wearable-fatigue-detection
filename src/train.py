# Exemple
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def train_model(model, X_train, y_train):
    """Entraîne le modèle"""
    return model.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test):
    """Calcule les métriques"""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
