from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

MESSAGES_TOTAL = Counter("spamdet_messages_total", "Total classified messages", ["label"])
CONFIDENCE = Histogram(
    "spamdet_confidence",
    "Prediction confidence distribution",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
REVIEW_QUEUE_TOTAL = Counter("spamdet_review_queue_total", "Total messages routed to human review")
OUTBREAK_ALERTS_TOTAL = Counter("spamdet_outbreak_alerts_total", "Total outbreak (near-duplicate) alerts raised")


def record_prediction(*, label: str, confidence: float, needs_review: bool, is_outbreak: bool) -> None:
    MESSAGES_TOTAL.labels(label=label).inc()
    CONFIDENCE.observe(confidence)
    if needs_review:
        REVIEW_QUEUE_TOTAL.inc()
    if is_outbreak:
        OUTBREAK_ALERTS_TOTAL.inc()


def render_latest() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
