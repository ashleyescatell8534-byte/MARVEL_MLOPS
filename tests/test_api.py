
from fastapi.testclient import TestClient
import app as api


def test_home_page(monkeypatch):
    """Comprueba que la página principal carga correctamente."""
    monkeypatch.setattr(api, "load_champion", lambda: object())

    with TestClient(api.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict_success(monkeypatch):
    """Comprueba que la API devuelve una predicción."""
    monkeypatch.setattr(api, "load_champion", lambda: object())
    monkeypatch.setattr(
        api,
        "predict_dialogue",
        lambda model, dialogue: "TONY STARK"
    )

    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            json={"dialogue": "I am Iron Man."}
        )

    assert response.status_code == 200
    assert response.json()["predicted_character"] == "TONY STARK"
    assert response.json()["alias"] == "champion"


def test_predict_empty_dialogue(monkeypatch):
    """Comprueba que la API rechaza un diálogo vacío."""
    monkeypatch.setattr(api, "load_champion", lambda: object())

    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            json={"dialogue": ""}
        )

    assert response.status_code == 422
