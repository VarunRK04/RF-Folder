import pytest

import app as app_module


@pytest.fixture
def client():
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as client:
        yield client


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_predict_missing_body(client):
    resp = client.post("/predict")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_missing_key(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Missing key: news_text"


def test_predict_non_string(client):
    resp = client.post("/predict", json={"news_text": 12345})
    assert resp.status_code == 400


def test_predict_non_object_body(client):
    resp = client.post("/predict", json=["not", "an", "object"])
    assert resp.status_code == 400


def test_predict_empty_string(client):
    resp = client.post("/predict", json={"news_text": "   "})
    assert resp.status_code == 400


def test_predict_too_long(client):
    resp = client.post("/predict", json={"news_text": "x" * (app_module.MAX_TEXT_LENGTH + 1)})
    assert resp.status_code == 400


def test_predict_valid(client):
    resp = client.post("/predict", json={"news_text": "This is a short news snippet about a local event."})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["prediction"] in ("FAKE", "REAL")
    assert "confidence" in body
    assert set(body["confidence"].keys()) == {"FAKE", "REAL"}
    assert abs(sum(body["confidence"].values()) - 1.0) < 1e-3
