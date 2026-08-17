import fakeredis
import pytest
from fastapi.testclient import TestClient

from spamdet.api.app import create_app
from spamdet.model.inference import PredictionResult


class _FakeClassifier:
    def __init__(self, results_by_text: dict[str, PredictionResult], default: PredictionResult | None = None):
        self._results = results_by_text
        self._default = default

    def predict(self, text: str) -> PredictionResult:
        return self._results.get(text, self._default)


HIGH_CONFIDENCE = PredictionResult(
    label="bilgilendirme", confidence=0.95, probabilities={"bilgilendirme": 0.95, "spam": 0.05}
)
BORDERLINE = PredictionResult(label="spam", confidence=0.5, probabilities={"spam": 0.5, "bilgilendirme": 0.5})


@pytest.fixture
def client():
    fake_classifier = _FakeClassifier({}, default=HIGH_CONFIDENCE)
    fake_redis = fakeredis.FakeStrictRedis(decode_responses=False)
    app = create_app(classifier=fake_classifier, redis_client=fake_redis)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_with_borderline():
    fake_classifier = _FakeClassifier({}, default=BORDERLINE)
    fake_redis = fakeredis.FakeStrictRedis(decode_responses=False)
    app = create_app(classifier=fake_classifier, redis_client=fake_redis)
    with TestClient(app) as test_client:
        yield test_client


def test_classify_otp_text_short_circuits_the_ml_classifier(client):
    resp = client.post("/classify", json={"text": "Kodunuz: 123456. Kimseyle paylasmayin."})
    body = resp.json()
    assert body["label"] == "otp"
    assert body["confidence"] == 0.95


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_classify_high_confidence_not_queued(client):
    resp = client.post("/classify", json={"text": "yarin gorusuruz"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "bilgilendirme"
    assert body["needs_review"] is False
    assert client.get("/review/pending").json() == []


def test_classify_generates_message_id_when_absent(client):
    resp = client.post("/classify", json={"text": "test"})
    assert resp.json()["message_id"]


def test_classify_respects_provided_message_id(client):
    resp = client.post("/classify", json={"text": "test", "message_id": "custom-id"})
    assert resp.json()["message_id"] == "custom-id"


def test_classify_borderline_confidence_queued_for_review(client_with_borderline):
    resp = client_with_borderline.post("/classify", json={"text": "belirsiz bir mesaj"})
    assert resp.json()["needs_review"] is True

    pending = client_with_borderline.get("/review/pending").json()
    assert len(pending) == 1
    assert pending[0]["text"] == "belirsiz bir mesaj"
    assert pending[0]["label"] == "spam"


def test_review_decide_approve_appends_confirmed_record(client_with_borderline, tmp_path, monkeypatch):
    confirmed_path = tmp_path / "confirmed.jsonl"
    monkeypatch.setenv("SPAMDET_CONFIRMED_DATA_PATH", str(confirmed_path))

    client_with_borderline.post("/classify", json={"text": "belirsiz mesaj", "message_id": "m1"})
    item_id = client_with_borderline.get("/review/pending").json()[0]["item_id"]

    resp = client_with_borderline.post(f"/review/{item_id}/decide", json={"decision": "approve"})
    assert resp.status_code == 200
    assert resp.json()["appended_to_training_data"] is True

    lines = confirmed_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "belirsiz mesaj" in lines[0]
    assert client_with_borderline.get("/review/pending").json() == []


def test_review_decide_reject_does_not_append(client_with_borderline, tmp_path, monkeypatch):
    confirmed_path = tmp_path / "confirmed.jsonl"
    monkeypatch.setenv("SPAMDET_CONFIRMED_DATA_PATH", str(confirmed_path))

    client_with_borderline.post("/classify", json={"text": "belirsiz mesaj", "message_id": "m1"})
    item_id = client_with_borderline.get("/review/pending").json()[0]["item_id"]

    resp = client_with_borderline.post(f"/review/{item_id}/decide", json={"decision": "reject"})
    assert resp.json()["appended_to_training_data"] is False
    assert not confirmed_path.exists()


def test_review_decide_unknown_item_returns_404(client):
    resp = client.post("/review/does-not-exist/decide", json={"decision": "approve"})
    assert resp.status_code == 404


def test_demo_page_renders(client):
    resp = client.get("/demo")
    assert resp.status_code == 200
    assert "Siniflandir" in resp.text
    assert "/classify" in resp.text


def test_dashboard_renders_pending_items(client_with_borderline):
    client_with_borderline.post("/classify", json={"text": "belirsiz mesaj", "message_id": "m1"})
    resp = client_with_borderline.get("/dashboard")
    assert resp.status_code == 200
    assert "belirsiz mesaj" in resp.text


def test_dashboard_shows_all_four_categories_in_correction_dropdown(client_with_borderline):
    client_with_borderline.post("/classify", json={"text": "belirsiz mesaj", "message_id": "m1"})
    resp = client_with_borderline.get("/dashboard")
    for label in ("otp", "reklam", "spam", "bilgilendirme"):
        assert f'value="{label}"' in resp.text


def test_dashboard_form_decide_redirects_and_resolves(client_with_borderline, tmp_path, monkeypatch):
    confirmed_path = tmp_path / "confirmed.jsonl"
    monkeypatch.setenv("SPAMDET_CONFIRMED_DATA_PATH", str(confirmed_path))

    client_with_borderline.post("/classify", json={"text": "belirsiz mesaj", "message_id": "m1"})
    item_id = client_with_borderline.get("/review/pending").json()[0]["item_id"]

    resp = client_with_borderline.post(
        f"/dashboard/review/{item_id}",
        data={"decision": "approve", "corrected_label": "spam"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/dashboard"
    assert confirmed_path.exists()


def test_metrics_endpoint(client):
    client.post("/classify", json={"text": "test"})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"spamdet_messages_total" in resp.content
