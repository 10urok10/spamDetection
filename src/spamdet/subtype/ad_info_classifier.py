from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

REKLAM = "reklam"
BILGILENDIRME = "bilgilendirme"


@dataclass(frozen=True)
class AdInfoPrediction:
    subtype: str  # "reklam" or "bilgilendirme"
    confidence: float  # probability of the predicted class
    reklam_probability: float  # P(reklam) regardless of which class won - useful for threshold inspection


class AdInfoClassifier:
    """Lightweight reklam-vs-bilgilendirme classifier for messages the
    top-level model already called 'legitimate' and the OTP rule
    (subtype.rules.detect_otp) didn't catch.

    TF-IDF + logistic regression rather than a full transformer
    fine-tune - deliberately the cheap option first (see docs/model.md):
    evaluate it, only escalate to a dedicated fine-tuned model if the
    numbers don't hold up.

    A Mersis-number/opt-out-phrase RULE was deliberately not used to
    gate this decision (unlike the OTP rule) - real data disproved that
    assumption: a genuine customer-satisfaction survey message contains
    both a Mersis number and a "RET yaz" opt-out phrase despite not
    being an ad (see the bilgilendirme entry notes in
    data/synthetic/seeds/legitimate.yaml). Those markers are regulated-
    bulk-SMS compliance signals in general, not ad-specific ones - so
    they're left as ordinary TF-IDF features for this classifier to
    weigh alongside promotional vocabulary, rather than hard-coded
    triggers.

    Recall on `reklam` is prioritized over precision (compliance-audit
    use case: missing a real ad matters more than double-checking a
    borderline informational message) via `reklam_threshold` - a single
    visible knob rather than folding the trade-off into class weights.
    """

    def __init__(self, *, reklam_threshold: float = 0.4):
        self.reklam_threshold = reklam_threshold
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
            ]
        )

    def fit(self, texts: list[str], labels: list[str]) -> None:
        self.pipeline.fit(texts, labels)

    def predict(self, text: str) -> AdInfoPrediction:
        classes = list(self.pipeline.named_steps["clf"].classes_)
        proba = self.pipeline.predict_proba([text])[0]
        reklam_proba = float(proba[classes.index(REKLAM)])

        if reklam_proba >= self.reklam_threshold:
            return AdInfoPrediction(subtype=REKLAM, confidence=reklam_proba, reklam_probability=reklam_proba)
        bilgilendirme_proba = float(proba[classes.index(BILGILENDIRME)])
        return AdInfoPrediction(
            subtype=BILGILENDIRME, confidence=bilgilendirme_proba, reklam_probability=reklam_proba
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "reklam_threshold": self.reklam_threshold}, path)

    @classmethod
    def load(cls, path: str | Path) -> "AdInfoClassifier":
        data = joblib.load(path)
        instance = cls(reklam_threshold=data["reklam_threshold"])
        instance.pipeline = data["pipeline"]
        return instance
