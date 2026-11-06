import optuna
from optuna_integration.mlflow import MLflowCallback

from src.data import load_data, split_data
from src.evaluate import evaluate_model
from src.train import train_model

STORAGE = "sqlite:///optuna.db"
STUDY_NAME = "klasyfikator-nowotworow"
TRACKING_URI = "http://127.0.0.1:5000"


def objective(trial, X_train, X_test, y_train, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 16),
    }
    model = train_model(X_train, y_train, params)
    return evaluate_model(model, X_test, y_test)


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    mlflow_callback = MLflowCallback(
        tracking_uri=TRACKING_URI,
        metric_name="accuracy",
    )

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: objective(trial, X_train, X_test, y_train, y_test),
        n_trials=50,
        callbacks=[mlflow_callback],
    )

    print(f"Najlepsze parametry: {study.best_params}")
    print(f"Najlepsza trafnosc: {study.best_value:.4f}")


if __name__ == "__main__":
    main()
