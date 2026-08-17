"""Standalone local tool for manually relabeling public-dataset rows
into otp/reklam/bilgilendirme/spam - see src/spamdet/manual_labels.py
for why this exists (short version: a manual sample check found the
"spam"-labeled bucket is overwhelmingly real advertisements, not fraud -
so it's reviewed first, "ham" second). Deliberately NOT part of the live
API (create_app()): no model/Redis/outbreak dependency, just a fast
local labeling loop.

Progress is saved incrementally to data/manual_labels/relabeled.jsonl as
you go, so it's always safe to stop (Ctrl+C) and resume later - already-
decided messages never resurface.

Usage:
    python scripts/label_tool.py
Then open http://localhost:8010/label in a browser. Click a category
button or press 1/2/3/4/0 on your keyboard (1=bilgilendirme, 2=otp,
3=reklam, 4=spam, 0=skip/unclear).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import uvicorn  # noqa: E402
from fastapi import FastAPI, Form, Request  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402

from spamdet.build_dataset import DEFAULT_RAW_DIR  # noqa: E402
from spamdet.manual_labels import (  # noqa: E402
    already_decided_texts,
    append_decision,
    build_candidate_pool,
    default_output_path,
)

TEMPLATES_DIR = PROJECT_ROOT / "src" / "spamdet" / "api" / "templates"
OUTPUT_PATH = default_output_path(PROJECT_ROOT)

app = FastAPI(title="spamdet label tool")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Built once at startup (not per-request - re-running the loaders on
# every page load would make each click noticeably slower for no
# reason). DONE is seeded from the output file so restarting the tool
# resumes exactly where you left off.
POOL: list[dict] = build_candidate_pool(DEFAULT_RAW_DIR)
DONE: set[str] = already_decided_texts(OUTPUT_PATH)


def _next_item() -> dict | None:
    for item in POOL:
        if item["text"] not in DONE:
            return item
    return None


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/label")


@app.get("/label")
def label_page(request: Request):
    item = _next_item()
    return templates.TemplateResponse(
        request,
        "label.html",
        {"item": item, "done": len(DONE), "total": len(POOL), "remaining": len(POOL) - len(DONE)},
    )


@app.post("/label/decide")
def label_decide(
    text: str = Form(...), original_source: str = Form(...), original_label: str = Form(...), label: str = Form(...)
) -> RedirectResponse:
    append_decision(OUTPUT_PATH, text=text, label=label, original_label=original_label, original_source=original_source)
    DONE.add(text)
    return RedirectResponse(url="/label", status_code=303)


if __name__ == "__main__":
    print(f"Havuz: {len(POOL)} mesaj, {len(DONE)} zaten etiketlenmis, {len(POOL) - len(DONE)} kaldi.")
    print(f"Kayit dosyasi: {OUTPUT_PATH}")
    print("Ac: http://localhost:8010/label")
    uvicorn.run(app, host="127.0.0.1", port=8010)
