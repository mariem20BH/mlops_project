"""
main.py
Point d'entrée : exécute les étapes du pipeline via des arguments CLI.
"""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML")
    parser.add_argument("--data", required=True, help="Fichier CSV")
    parser.add_argument("--target", required=True, help="Nom de la colonne cible")
    parser.add_argument(
        "--step",
        choices=["prepare", "train", "evaluate", "all"],
        default="all",
        help="Étape à exécuter",
    )
    parser.add_argument("--model", default="model.joblib", help="Chemin du modèle")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, _ = prepare_data(args.data, args.target)

    if args.step == "prepare":
        print(f"Données prêtes : train={X_train.shape}, test={X_test.shape}")

    elif args.step == "train":
        model = train_model(X_train, y_train)
        save_model(model, args.model)

    elif args.step == "evaluate":
        model = load_model(args.model)
        evaluate_model(model, X_test, y_test)

    else:
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        save_model(model, args.model)


if __name__ == "__main__":
    main()
