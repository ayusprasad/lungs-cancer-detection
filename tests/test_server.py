import server


def test_int_value_clamps_and_uses_default_for_invalid_input():
    assert server._int_value({"epochs": "100"}, "epochs", 8, 1, 50) == 50
    assert server._int_value({"epochs": "invalid"}, "epochs", 8, 1, 50) == 8


def test_tabular_prediction_fails_gracefully_without_a_trained_model(monkeypatch):
    monkeypatch.setattr(server, "load_tabular_model", lambda: None)
    response = server.app.test_client().post("/api/predict/tabular", json={})

    assert response.status_code == 409
    assert response.get_json()["ok"] is False
    assert "Train the tabular model" in response.get_json()["error"]


def test_image_prediction_fails_gracefully_without_a_trained_model(monkeypatch):
    monkeypatch.setattr(server, "load_image_model", lambda: None)
    response = server.app.test_client().post("/api/predict/image")

    assert response.status_code == 409
    assert response.get_json()["ok"] is False
    assert "Train the image model" in response.get_json()["error"]


def test_tabular_prediction_reports_missing_fields(monkeypatch):
    monkeypatch.setattr(server, "load_tabular_model", lambda: {"model": object()})
    response = server.app.test_client().post("/api/predict/tabular", json={})

    assert response.status_code == 400
    assert response.get_json()["ok"] is False
    assert "Missing tabular fields" in response.get_json()["error"]
