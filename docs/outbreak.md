# Outbreak detection (Stage 2)

## What's implemented: SimHash + Redis LSH-banding (primary layer)

`src/spamdet/outbreak/`:

- `simhash.py` - 64-bit SimHash fingerprint over character 4-shingles
  (pure stdlib, `hashlib.blake2b` for stable cross-process hashing - not
  Python's built-in `hash()`, which is randomized per-process).
- `lsh.py` - `RedisLSHIndex`: splits each fingerprint into bands and
  indexes each band in its own Redis set, so finding near-duplicate
  candidates is O(num_bands) Redis lookups instead of comparing against
  every stored message.
- `detector.py` - `OutbreakDetector.ingest(message_id, text)`: cleans text
  through Stage 1's `strip_zero_width`/`strip_confusables` first (so
  evasion-corrupted text-spun copies still hash close together), computes
  the fingerprint, finds LSH candidates, confirms with an exact
  Hamming-similarity check, and returns which prior messages (if any) it
  matches.

This runs on every ingested message (real-time), per the project's design
decision to keep the expensive clustering step off the hot path.

### Tuning note: band width is a recall/precision knob

LSH banding only *proposes candidates* - the real decision is the
Hamming-similarity threshold check that follows. That means it's safe to
prefer more/narrower bands (higher candidate recall, e.g. 8 bands x 8 bits
for a 64-bit fingerprint, the default here) over fewer/wider bands: a
false-positive candidate just costs one extra comparison, while too few
bands can miss real near-duplicates whose small number of differing bits
happen to land one-per-band. This was tuned empirically during development
- see git history on `src/spamdet/outbreak/lsh.py` if the default ever
needs revisiting.

`similarity_threshold` (default 0.90 on `OutbreakDetector`, used at 0.85 in
the live smoke test) is a genuine tuning knob: two messages differing by a
whole reworded clause (not just a swapped link/name/number) will legitimately
score lower and may not count as "the same outbreak" - that's the threshold
doing its job, not a bug.

### Running it

```
docker compose up -d redis
```

Then use `OutbreakDetector(redis.Redis(host="localhost", port=6379))`
(see `src/spamdet/outbreak/detector.py`). Tests use `fakeredis` and never
require a real Redis instance; the docker-compose Redis is only for actual
runtime / manual verification.

## What's NOT implemented (documented per project scope, not built)

Per the project's explicit design decisions, these are out of Stage 2
scope and deferred:

- **SBERT + vector DB secondary layer**: only meant to re-check messages
  the SimHash layer flags as "suspicious but not confident" - SimHash/LSH
  alone is the priority and is sufficient for the MVP.
- **Periodic batch clustering (HDBSCAN)**: intended to run on a schedule
  (e.g. every 5 minutes) over accumulated messages for broader
  campaign-level clustering, not per-message. Not implemented; would be a
  separate scheduled job reading from the same Redis fingerprint store.

Both remain future-work notes per the project brief's instruction not to
expand MVP scope, not oversights.
