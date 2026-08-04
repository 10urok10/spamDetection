from spamdet.api import metrics


def test_record_prediction_increments_counters():
    before = metrics.MESSAGES_TOTAL.labels(label="phishing")._value.get()
    before_review = metrics.REVIEW_QUEUE_TOTAL._value.get()
    before_outbreak = metrics.OUTBREAK_ALERTS_TOTAL._value.get()

    metrics.record_prediction(label="phishing", confidence=0.5, needs_review=True, is_outbreak=True)

    assert metrics.MESSAGES_TOTAL.labels(label="phishing")._value.get() == before + 1
    assert metrics.REVIEW_QUEUE_TOTAL._value.get() == before_review + 1
    assert metrics.OUTBREAK_ALERTS_TOTAL._value.get() == before_outbreak + 1


def test_record_prediction_does_not_increment_review_or_outbreak_when_false():
    before_review = metrics.REVIEW_QUEUE_TOTAL._value.get()
    before_outbreak = metrics.OUTBREAK_ALERTS_TOTAL._value.get()

    metrics.record_prediction(label="legitimate", confidence=0.95, needs_review=False, is_outbreak=False)

    assert metrics.REVIEW_QUEUE_TOTAL._value.get() == before_review
    assert metrics.OUTBREAK_ALERTS_TOTAL._value.get() == before_outbreak


def test_render_latest_includes_metric_names():
    metrics.record_prediction(label="spam", confidence=0.9, needs_review=False, is_outbreak=False)
    payload, content_type = metrics.render_latest()
    assert b"spamdet_messages_total" in payload
    assert "text/plain" in content_type
