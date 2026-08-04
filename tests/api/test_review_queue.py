import fakeredis
import pytest

from spamdet.api.review_queue import ReviewQueue


@pytest.fixture
def queue():
    return ReviewQueue(fakeredis.FakeStrictRedis(decode_responses=False))


def test_enqueue_and_list_pending(queue):
    item_id = queue.enqueue(
        message_id="m1", text="orijinal metin", cleaned_text="orijinal metin", label="phishing", confidence=0.5,
        probabilities={"phishing": 0.5, "legitimate": 0.5},
    )
    pending = queue.list_pending()
    assert len(pending) == 1
    assert pending[0].item_id == item_id
    assert pending[0].message_id == "m1"
    assert pending[0].label == "phishing"
    assert pending[0].probabilities == {"phishing": 0.5, "legitimate": 0.5}


def test_list_pending_is_fifo_ordered(queue):
    id1 = queue.enqueue(message_id="m1", text="a", cleaned_text="a", label="spam", confidence=0.5, probabilities={})
    id2 = queue.enqueue(message_id="m2", text="b", cleaned_text="b", label="spam", confidence=0.5, probabilities={})
    pending = queue.list_pending()
    assert [item.item_id for item in pending] == [id1, id2]


def test_resolve_removes_item_from_pending(queue):
    item_id = queue.enqueue(
        message_id="m1", text="a", cleaned_text="a", label="spam", confidence=0.5, probabilities={}
    )
    resolved = queue.resolve(item_id)
    assert resolved.item_id == item_id
    assert queue.list_pending() == []


def test_resolve_unknown_item_returns_none(queue):
    assert queue.resolve("does-not-exist") is None


def test_list_pending_respects_limit(queue):
    for i in range(5):
        queue.enqueue(message_id=f"m{i}", text="x", cleaned_text="x", label="spam", confidence=0.5, probabilities={})
    assert len(queue.list_pending(limit=2)) == 2
