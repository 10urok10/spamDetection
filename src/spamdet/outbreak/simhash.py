import hashlib


def _shingle(text: str, shingle_size: int) -> set[str]:
    normalized = " ".join(text.split()).lower()
    if len(normalized) <= shingle_size:
        return {normalized} if normalized else set()
    return {normalized[i : i + shingle_size] for i in range(len(normalized) - shingle_size + 1)}


def _stable_hash(token: str, hash_bits: int) -> int:
    # hashlib (not builtin hash()) so fingerprints are reproducible across
    # processes/machines - str hashing is randomized per-process by default.
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=hash_bits // 8).digest()
    return int.from_bytes(digest, "big")


def simhash(text: str, *, hash_bits: int = 64, shingle_size: int = 4) -> int:
    """Compute a SimHash fingerprint: near-duplicate texts (e.g.
    lightly text-spun copies of the same spam blast) produce fingerprints
    with a small Hamming distance, letting outbreak detection work off
    approximate similarity instead of exact-text matching.
    """
    shingles = _shingle(text, shingle_size)
    if not shingles:
        return 0

    weights = [0] * hash_bits
    for shingle in shingles:
        h = _stable_hash(shingle, hash_bits)
        for bit in range(hash_bits):
            weights[bit] += 1 if (h >> bit) & 1 else -1

    fingerprint = 0
    for bit in range(hash_bits):
        if weights[bit] > 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def similarity(a: int, b: int, *, hash_bits: int = 64) -> float:
    """1.0 = identical fingerprints, 0.0 = maximally different."""
    return 1.0 - hamming_distance(a, b) / hash_bits
