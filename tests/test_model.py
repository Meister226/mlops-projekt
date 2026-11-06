import numpy as np
import pytest

from src.data import load_data, split_data
from src.evaluate import evaluate_model
from src.train import train_model


def test_load_data():
    """Test danych: wymiary, kompletnosc i liczba klas."""
    X, y = load_data()
    # Liczba obserwacji w X i y musi byc zgodna
    assert X.shape[0] == y.shape[0]
    # Zbior nie moze zawierac wartosci pustych
    assert not np.isnan(X).any()
    # Problem jest klasyfikacja binarna, oczekiwane sa dwie klasy
    assert len(np.unique(y)) == 2


def test_ksztalt_predykcji():
    """Test kontraktu: ksztalt i dziedzina predykcji."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    predykcje = model.predict(X_test)
    # Liczba predykcji musi odpowiadac liczbie obserwacji testowych
    assert predykcje.shape[0] == X_test.shape[0]
    # Predykcje moga przyjmowac wylacznie wartosci etykiet klas
    assert set(np.unique(predykcje)).issubset({0, 1})


def test_prog_trafnosci():
    """Test walidacyjny: minimalna akceptowalna jakosc modelu."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    # Model gorszy od progu oznacza regresje jakosci w projekcie
    assert accuracy > 0.90


def test_split_data_sizes_and_proportions():
    """Test licznosci podzbiorow oraz proporcji podzialu z parametrem test_size"""
    X, y = load_data()
    test_size = 0.25
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state=42)

    assert len(X_train) + len(X_test) == len(X)

    assert pytest.approx(len(X_test) / len(X), rel=0.01) == test_size

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_split_data_stratification():
    """2. Test weryfikujący zachowanie proporcji klas po podziale stratyfikowanym."""
    X, y = load_data()

    # Obliczamy początkowe proporcje klas w całej populacji
    classes, counts = np.unique(y, return_counts=True)
    original_proportions = counts / len(y)

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=42)

    train_classes, train_counts = np.unique(y_train, return_counts=True)
    train_proportions = train_counts / len(y_train)

    test_classes, test_counts = np.unique(y_test, return_counts=True)
    test_proportions = test_counts / len(y_test)

    np.testing.assert_allclose(train_proportions, original_proportions, atol=0.05)
    np.testing.assert_allclose(test_proportions, original_proportions, atol=0.05)
