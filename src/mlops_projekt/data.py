from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_data():
    dane = load_breast_cancer()
    return dane.data, dane.target


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
