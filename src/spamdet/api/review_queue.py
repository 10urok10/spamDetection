import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..outbreak.lsh import RedisLike


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


@dataclass(frozen=True)
class ReviewItem:
    item_id: str
    message_id: str
    text: str
    cleaned_text: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    created_at: float


class ReviewQueue:
    """Redis-backed queue for messages whose classification confidence
    fell in the human-review band. A sorted set (score = enqueue time)
    gives FIFO ordering for the dashboard; each item's full payload lives
    in its own hash.
    """

    def __init__(self, redis_client: RedisLike, *, key_prefix: str = "spamdet:review"):
        self.redis = redis_client
        self.key_prefix = key_prefix

    def _item_key(self, item_id: str) -> str:
        return f"{self.key_prefix}:item:{item_id}"

    def _pending_key(self) -> str:
        return f"{self.key_prefix}:pending"

    def enqueue(
        self,
        *,
        message_id: str,
        text: str,
        cleaned_text: str,
        label: str,
        confidence: float,
        probabilities: dict[str, float],
    ) -> str:
        item_id = str(uuid.uuid4())
        created_at = time.time()
        payload = {
            "item_id": item_id,
            "message_id": message_id,
            "text": text,
            "cleaned_text": cleaned_text,
            "label": label,
            "confidence": confidence,
            "probabilities": json.dumps(probabilities, ensure_ascii=False),
            "created_at": created_at,
        }
        pipe = self.redis.pipeline()
        pipe.hset(self._item_key(item_id), mapping=payload)
        pipe.zadd(self._pending_key(), {item_id: created_at})
        pipe.execute()
        return item_id

    def _load(self, item_id: str) -> ReviewItem | None:
        raw = self.redis.hgetall(self._item_key(item_id))
        if not raw:
            return None
        data = {_decode(k): _decode(v) for k, v in raw.items()}
        return ReviewItem(
            item_id=data["item_id"],
            message_id=data["message_id"],
            text=data["text"],
            cleaned_text=data["cleaned_text"],
            label=data["label"],
            confidence=float(data["confidence"]),
            probabilities=json.loads(data["probabilities"]),
            created_at=float(data["created_at"]),
        )

    def list_pending(self, *, limit: int = 50) -> list[ReviewItem]:
        ids = [_decode(i) for i in self.redis.zrange(self._pending_key(), 0, limit - 1)]
        items = []
        for item_id in ids:
            item = self._load(item_id)
            if item is not None:
                items.append(item)
        return items

    def resolve(self, item_id: str) -> ReviewItem | None:
        """Remove an item from the pending queue and return what it was,
        so the caller (the API route) can decide what to do with the
        decision - e.g. append an approved/corrected label to the
        retraining data file.
        """
        item = self._load(item_id)
        if item is None:
            return None
        pipe = self.redis.pipeline()
        pipe.zrem(self._pending_key(), item_id)
        pipe.delete(self._item_key(item_id))
        pipe.execute()
        return item


def append_confirmed_record(path: Path, *, text: str, label: str, lang: str = "tr") -> None:
    """Persist a human-approved (possibly relabeled) review item in the
    same {text,label,source,lang} shape as Stage 1's Record, so it can
    later be folded into retraining data alongside the other sources.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"text": text, "label": label, "source": "human_reviewed", "lang": lang}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
