"""Tests unitaires de model_pipeline.py"""

import numpy as np
import pandas as pd

from model_pipeline import load_model, prepare_data, save_model, train_model


def make_csv(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=100),
            "b": rng.normal(size=100),
            "cat": rng.choice(["x", "y"], 100),
            "income": rng.choice(["<=50K", ">50K"], 100),
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_prepare_data(tmp_path):
    X_train, X_test, y_train, y_test, _ = prepare_data(make_csv(tmp_path), "income")
    assert len(X_train) == 80
    assert len(X_test) == 20


def test_train_save_load(tmp_path):
    X_train, X_test, y_train, y_test, _ = prepare_data(make_csv(tmp_path), "income")
    model = train_model(X_train, y_train, n_estimators=10)
    path = str(tmp_path / "m.joblib")
    save_model(model, path)
    loaded = load_model(path)
    assert (loaded.predict(X_test) == model.predict(X_test)).all()
