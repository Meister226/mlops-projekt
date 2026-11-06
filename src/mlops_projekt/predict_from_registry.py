import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model = mlflow.sklearn.load_model("models:/klasyfikator-nowotworow@champion")

X, _ = load_breast_cancer(return_X_y=True)
predictions = model.predict(X[:20])
print(f"Predykcje dla pierwszych dziesieciu próbek: {predictions}")
