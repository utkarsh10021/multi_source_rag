from fastapi.testclient import TestClient
import api

def test_health(monkeypatch):
    class FakeStore:
        vectorstore = None
        chunks = []

    class FakeService:
        store = FakeStore()

    monkeypatch.setattr(api, "service", FakeService())
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
