from sklearn.metrics import accuracy_score


def evaluate_model(model, X_test, y_test):
    predykcje = model.predict(X_test)
    return accuracy_score(y_test, predykcje)
