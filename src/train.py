import mlflow
import optuna
from sklearn.ensemble import RandomForestClassifier

from src.data import load_data, split_data
from src.evaluate import evaluate_model

STORAGE = "sqlite:///optuna.db"
STUDY_NAME = "klasyfikator-nowotworow"
TRACKING_URI = "http://127.0.0.1:5000"
NAZWA_MODELU = "klasyfikator-nowotworow"

DOMYSLNE_PARAMETRY = {"n_estimators": 100, "max_depth": 5}


def get_best_params():
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
        return study.best_params
    except KeyError:
        print("Studium Optuny nie istnieje, uzyto parametrowdomyslnych.")
    return DOMYSLNE_PARAMETRY


def train_model(X_train, y_train, params=None):
    if params is None:
        params = DOMYSLNE_PARAMETRY
    model = RandomForestClassifier(random_state=42, **params)
    model.fit(X_train, y_train)
    return model


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    params = get_best_params()
    model = train_model(X_train, y_train, params)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Trafnosc modelu na zbiorze testowym: {accuracy:.4f}")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("trening-finalny")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        sygnatura = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=sygnatura,
            registered_model_name=NAZWA_MODELU,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )


if __name__ == "__main__":
    main()
