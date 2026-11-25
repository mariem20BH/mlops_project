"""
pipeline_prefect.py
Pipeline Prefect : tasks et flows (install, code, train, evaluate, all).
"""

import argparse
import subprocess  # nosec B404
import sys

from prefect import flow, get_run_logger, task

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

DATA = "adult_clean.csv"
TARGET = "income"
MODEL_PATH = "model.joblib"
# Seulement nos fichiers (sinon l'analyse scanne tout le venv)
FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run_cmd(cmd, check=True):
    """Exécute une commande shell et affiche sa sortie dans les logs Prefect."""
    logger = get_run_logger()
    logger.info("Commande : %s", " ".join(cmd))
    result = subprocess.run(  # nosec B603
        cmd, capture_output=True, text=True, check=False
    )
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Échec de la commande : {' '.join(cmd)}")
    return result.returncode


# ---------------- TASKS ----------------
@task(name="Installer les dépendances")
def install_dependencies():
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="Formatage du code")
def format_code():
    run_cmd([sys.executable, "-m", "black", *FILES])


@task(name="Qualité du code")
def check_quality():
    run_cmd(
        [sys.executable, "-m", "flake8", "--max-line-length=100", *FILES],
        check=False,
    )


@task(name="Sécurité du code")
def check_security():
    run_cmd([sys.executable, "-m", "bandit", "-q", *FILES], check=False)


@task(name="Tests unitaires")
def run_tests():
    run_cmd([sys.executable, "-m", "pytest", "-q", "test_model_pipeline.py"])


@task(name="Préparation des données")
def prepare_task(data_path, target):
    return prepare_data(data_path, target)


@task(name="Entraînement du modèle")
def train_task(X_train, y_train):
    return train_model(X_train, y_train)


@task(name="Sauvegarder le modèle")
def save_task(model, model_path):
    save_model(model, model_path)


@task(name="Charger le modèle")
def load_task(model_path):
    return load_model(model_path)


@task(name="Évaluer le modèle")
def evaluate_task(model, X_test, y_test):
    return evaluate_model(model, X_test, y_test)


# ---------------- FLOWS ----------------
@flow(name="install")
def install_flow():
    install_dependencies()


@flow(name="code")
def code_flow():
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_tests()


@flow(name="train")
def train_flow(data_path=DATA, target=TARGET, model_path=MODEL_PATH):
    X_train, X_test, y_train, y_test, _ = prepare_task(data_path, target)
    model = train_task(X_train, y_train)
    save_task(model, model_path)


@flow(name="evaluate")
def evaluate_flow(data_path=DATA, target=TARGET, model_path=MODEL_PATH):
    X_train, X_test, y_train, y_test, _ = prepare_task(data_path, target)
    model = load_task(model_path)
    return evaluate_task(model, X_test, y_test)


@flow(name="all")
def all_flow(data_path=DATA, target=TARGET, model_path=MODEL_PATH):
    # Étapes liées au code, avant la préparation des données
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_tests()
    # Étapes 2-3-4-6
    X_train, X_test, y_train, y_test, _ = prepare_task(data_path, target)
    model = train_task(X_train, y_train)
    save_task(model, model_path)
    return evaluate_task(model, X_test, y_test)


FLOWS = {
    "all": all_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
    "code": code_flow,
    "install": install_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect")
    parser.add_argument("--flow", choices=list(FLOWS), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
