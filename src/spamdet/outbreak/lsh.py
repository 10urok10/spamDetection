from typing import Protocol


class RedisLike(Protocol):
    """The subset of the redis-py client API this module needs - lets
    tests inject fakeredis (or any compatible client) without depending on
    the real `redis` package's types.
    """

    def pipeline(self): ...
    def smembers(self, key: str): ...
    def get(self, key: str): ...


class RedisLSHIndex:
    """SimHash + LSH-banding index on Redis: splits each fingerprint into
    ``num_bands`` chunks and indexes each chunk in its own Redis set, so
    finding near-duplicate candidates for a new message is O(num_bands)
    Redis lookups instead of comparing against every stored fingerprint.

    ``candidates()`` only proposes candidates (messages sharing at least
    one band) - it is not the similarity decision itself, callers still
    confirm with an exact Hamming-distance/similarity check afterward.
    That's why narrower bands (more bands, fewer bits each) are safe to
    prefer for recall: a false-positive candidate just costs one extra
    Hamming comparison, while too few/wide bands can miss real
    near-duplicates whose differing bits happen to be spread one-per-band.
    """

    def __init__(
        self,
        redis_client: RedisLike,
        *,
        num_bands: int = 8,
        hash_bits: int = 64,
        key_prefix: str = "spamdet:simhash",
    ):
        if hash_bits % num_bands != 0:
            raise ValueError(f"hash_bits ({hash_bits}) must be divisible by num_bands ({num_bands})")
        self.redis = redis_client
        self.num_bands = num_bands
        self.hash_bits = hash_bits
        self.band_bits = hash_bits // num_bands
        self.key_prefix = key_prefix

    def _band_values(self, fingerprint: int) -> list[int]:
        mask = (1 << self.band_bits) - 1
        return [(fingerprint >> (i * self.band_bits)) & mask for i in range(self.num_bands)]

    def _band_key(self, band_index: int, band_value: int) -> str:
        return f"{self.key_prefix}:band:{band_index}:{band_value}"

    def _fingerprint_key(self, message_id: str) -> str:
        return f"{self.key_prefix}:fp:{message_id}"

    def add(self, message_id: str, fingerprint: int, *, ttl_seconds: int | None = None) -> None:
        pipe = self.redis.pipeline()
        pipe.set(self._fingerprint_key(message_id), fingerprint, ex=ttl_seconds)
        for band_index, band_value in enumerate(self._band_values(fingerprint)):
            band_key = self._band_key(band_index, band_value)
            pipe.sadd(band_key, message_id)
            if ttl_seconds is not None:
                pipe.expire(band_key, ttl_seconds)
        pipe.execute()

    def candidates(self, fingerprint: int) -> set[str]:
        """Message ids sharing at least one band with ``fingerprint`` -
        candidates for a full Hamming-distance comparison, not confirmed
        near-duplicates yet.
        """
        ids: set[str] = set()
        for band_index, band_value in enumerate(self._band_values(fingerprint)):
            members = self.redis.smembers(self._band_key(band_index, band_value))
            ids |= {m.decode() if isinstance(m, bytes) else m for m in members}
        return ids

    def get_fingerprint(self, message_id: str) -> int | None:
        value = self.redis.get(self._fingerprint_key(message_id))
        if value is None:
            return None
        return int(value)
