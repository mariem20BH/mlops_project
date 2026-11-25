"""
model_pipeline.py
Fonctions modulaires pour le pipeline ML (générique : fonctionne avec n'importe quel CSV).
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(csv_path, target, test_size=0.2, random_state=42):
    """Charge et prétraite les données. Retourne X_train, X_test, y_train, y_test, scaler."""
    df = pd.read_csv(csv_path)

    if target not in df.columns:
        raise ValueError(
            f"Colonne cible '{target}' introuvable. Colonnes disponibles : {list(df.columns)}"
        )

    df = df.dropna()

    y = df[target]
    X = df.drop(columns=[target])

    if y.dtype == "object" or str(y.dtype) == "category" or y.dtype == "bool":
        y = pd.factorize(y)[0]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un modèle RandomForest."""
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Évalue les performances du modèle et retourne un dict de métriques."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    print("=== Évaluation du modèle ===")
    for name, value in metrics.items():
        print(f"{name:>10} : {value:.4f}")
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metrics


def save_model(model, path="model.joblib"):
    """Sauvegarde le modèle entraîné avec joblib."""
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans : {path}")


def load_model(path="model.joblib"):
    """Charge un modèle sauvegardé avec joblib."""
    model = joblib.load(path)
    print(f"Modèle chargé depuis : {path}")
    return model
