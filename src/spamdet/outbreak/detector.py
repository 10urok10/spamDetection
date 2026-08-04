from dataclasses import dataclass, field

from ..preprocessing.homoglyphs import strip_confusables
from ..preprocessing.zero_width import strip_zero_width
from .lsh import RedisLike, RedisLSHIndex
from .simhash import simhash, similarity


@dataclass(frozen=True)
class OutbreakResult:
    message_id: str
    fingerprint: int
    is_outbreak_candidate: bool
    similar_message_ids: list[str] = field(default_factory=list)
    similarities: dict[str, float] = field(default_factory=dict)


class OutbreakDetector:
    """Real-time (per-message) near-duplicate detection layer: SimHash +
    Redis LSH-banding, the primary/cheap outbreak signal. Runs on every
    ingested message - unlike the (not implemented in this stage, see
    docs/outbreak.md) periodic-batch SBERT+HDBSCAN secondary layer, which
    is only meant to re-check messages this layer flags as borderline.
    """

    def __init__(
        self,
        redis_client: RedisLike,
        *,
        num_bands: int = 8,
        hash_bits: int = 64,
        shingle_size: int = 4,
        similarity_threshold: float = 0.90,
        ttl_seconds: int | None = None,
    ):
        self.index = RedisLSHIndex(redis_client, num_bands=num_bands, hash_bits=hash_bits)
        self.hash_bits = hash_bits
        self.shingle_size = shingle_size
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def clean_text(text: str) -> str:
        """Undo the same adversarial surface manipulations Stage 1's
        preprocessing targets before hashing, so text-spun variants that
        also use zero-width/homoglyph evasion still land on a similar
        fingerprint instead of looking artificially distinct.
        """
        return strip_confusables(strip_zero_width(text))

    def ingest(self, message_id: str, text: str) -> OutbreakResult:
        cleaned = self.clean_text(text)
        fingerprint = simhash(cleaned, hash_bits=self.hash_bits, shingle_size=self.shingle_size)

        matches: list[tuple[str, float]] = []
        for candidate_id in self.index.candidates(fingerprint) - {message_id}:
            candidate_fp = self.index.get_fingerprint(candidate_id)
            if candidate_fp is None:
                continue
            score = similarity(fingerprint, candidate_fp, hash_bits=self.hash_bits)
            if score >= self.similarity_threshold:
                matches.append((candidate_id, score))
        matches.sort(key=lambda pair: pair[1], reverse=True)

        self.index.add(message_id, fingerprint, ttl_seconds=self.ttl_seconds)

        return OutbreakResult(
            message_id=message_id,
            fingerprint=fingerprint,
            is_outbreak_candidate=bool(matches),
            similar_message_ids=[m[0] for m in matches],
            similarities=dict(matches),
        )
