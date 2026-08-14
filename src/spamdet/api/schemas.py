from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1)
    message_id: str | None = None


class OutbreakInfo(BaseModel):
    is_outbreak_candidate: bool
    similar_message_ids: list[str]


class SubtypeInfo(BaseModel):
    subtype: str  # "otp" | "bilgilendirme" | "reklam"
    source: str  # "rule_otp" | "model"
    reklam_probability: float | None = None


class ClassifyResponse(BaseModel):
    message_id: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    cleaned_text: str
    homoglyph_attack_detected: bool
    urls_found: list[str]
    needs_review: bool
    outbreak: OutbreakInfo
    subtype: SubtypeInfo | None = None


class ReviewDecisionRequest(BaseModel):
    decision: str = Field(..., pattern="^(approve|reject)$")
    corrected_label: str | None = None


class ReviewItemResponse(BaseModel):
    item_id: str
    message_id: str
    text: str
    cleaned_text: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    created_at: float


class ReviewDecisionResponse(BaseModel):
    item_id: str
    resolved: bool
    appended_to_training_data: bool
