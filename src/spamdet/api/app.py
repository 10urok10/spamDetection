import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from ..model.inference import SpamClassifier
from ..model.labels import LABELS
from ..outbreak.detector import OutbreakDetector
from . import metrics
from .config import get_confirmed_data_path, get_model_dir, get_redis_url
from .pipeline import ClassificationPipeline
from .review_queue import ReviewQueue, append_confirmed_record
from .schemas import (
    ClassifyRequest,
    ClassifyResponse,
    OutbreakInfo,
    ReviewDecisionRequest,
    ReviewDecisionResponse,
    ReviewItemResponse,
)

TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app(*, classifier: SpamClassifier | None = None, redis_client=None) -> FastAPI:
    """Build the spamdet FastAPI app. ``classifier``/``redis_client`` are
    injectable so tests (and any embedding code) can supply fakes; when
    omitted, the lifespan handler constructs real ones from env vars
    (see api/config.py) on startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = classifier or SpamClassifier(get_model_dir())
        app.state.redis_client = redis_client
        if app.state.redis_client is None:
            import redis as redis_lib

            app.state.redis_client = redis_lib.Redis.from_url(get_redis_url())
        app.state.outbreak_detector = OutbreakDetector(app.state.redis_client)
        app.state.review_queue = ReviewQueue(app.state.redis_client)
        app.state.pipeline = ClassificationPipeline(app.state.classifier, app.state.outbreak_detector)
        yield

    app = FastAPI(title="spamdet API", version="0.1.0", lifespan=lifespan)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/classify", response_model=ClassifyResponse)
    def classify(payload: ClassifyRequest, request: Request) -> ClassifyResponse:
        pipeline: ClassificationPipeline = request.app.state.pipeline
        review_queue: ReviewQueue = request.app.state.review_queue
        message_id = payload.message_id or str(uuid.uuid4())

        result = pipeline.process(message_id, payload.text)
        is_outbreak = bool(result.outbreak and result.outbreak.is_outbreak_candidate)

        metrics.record_prediction(
            label=result.prediction.label,
            confidence=result.prediction.confidence,
            needs_review=result.needs_review,
            is_outbreak=is_outbreak,
        )

        if result.needs_review:
            review_queue.enqueue(
                message_id=message_id,
                text=result.text,
                cleaned_text=result.cleaned_text,
                label=result.prediction.label,
                confidence=result.prediction.confidence,
                probabilities=result.prediction.probabilities,
            )

        return ClassifyResponse(
            message_id=message_id,
            label=result.prediction.label,
            confidence=result.prediction.confidence,
            probabilities=result.prediction.probabilities,
            cleaned_text=result.cleaned_text,
            homoglyph_attack_detected=result.homoglyph_attack_detected,
            urls_found=result.urls_found,
            needs_review=result.needs_review,
            outbreak=OutbreakInfo(
                is_outbreak_candidate=is_outbreak,
                similar_message_ids=result.outbreak.similar_message_ids if result.outbreak else [],
            ),
        )

    @app.get("/review/pending", response_model=list[ReviewItemResponse])
    def list_pending_reviews(request: Request, limit: int = 50) -> list[ReviewItemResponse]:
        review_queue: ReviewQueue = request.app.state.review_queue
        return [ReviewItemResponse(**item.__dict__) for item in review_queue.list_pending(limit=limit)]

    @app.post("/review/{item_id}/decide", response_model=ReviewDecisionResponse)
    def decide_review(item_id: str, payload: ReviewDecisionRequest, request: Request) -> ReviewDecisionResponse:
        review_queue: ReviewQueue = request.app.state.review_queue
        item = review_queue.resolve(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="review item not found")

        appended = False
        if payload.decision == "approve":
            label = payload.corrected_label or item.label
            append_confirmed_record(get_confirmed_data_path(), text=item.text, label=label)
            appended = True

        return ReviewDecisionResponse(item_id=item_id, resolved=True, appended_to_training_data=appended)

    @app.get("/demo", response_class=HTMLResponse)
    def demo(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "demo.html", {})

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        review_queue: ReviewQueue = request.app.state.review_queue
        items = review_queue.list_pending(limit=100)
        return templates.TemplateResponse(
            request, "dashboard.html", {"items": items, "labels": [label.value for label in LABELS]}
        )

    @app.post("/dashboard/review/{item_id}")
    def dashboard_decide(
        item_id: str, request: Request, decision: str = Form(...), corrected_label: str = Form("")
    ) -> RedirectResponse:
        review_queue: ReviewQueue = request.app.state.review_queue
        item = review_queue.resolve(item_id)
        if item is not None and decision == "approve":
            label = corrected_label or item.label
            append_confirmed_record(get_confirmed_data_path(), text=item.text, label=label)
        return RedirectResponse(url="/dashboard", status_code=303)

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        payload, content_type = metrics.render_latest()
        return Response(content=payload, media_type=content_type)

    return app
