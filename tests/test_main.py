from fastapi.testclient import TestClient

from pokeragent.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_demo_page_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "Poker Hand Review Agent" in res.text


def test_review_rejects_bad_mode_without_calling_agent():
    res = client.post("/review", json={"hand_history": "x", "mode": "bogus"})
    assert res.status_code == 422
