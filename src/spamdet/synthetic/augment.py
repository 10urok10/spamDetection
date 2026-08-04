import random
import zlib
from typing import Protocol

from ..schema import Record

SOURCE_NAME = "synthetic_augmented"

# Deterministic, offline, rule-based paraphrasing: synonym-slot substitution
# for common spam/fraud trigger words + light surface jitter. Kept
# rule-based in Stage 1 to avoid API keys/network calls in the pipeline
# the tests exercise; an LLM-backed Paraphraser can be swapped in later
# without changing augment_examples()'s signature.
SYNONYM_MAP: dict[str, list[str]] = {
    "tiklayin": ["dokunun", "basin", "giris yapin"],
    "hemen": ["simdi", "aninda", "gecikmeden"],
    "bonus": ["hediye", "odul", "kazanc"],
    "kazandiniz": ["kazandin", "odul kazandiniz", "hak kazandiniz"],
    "hesabiniz": ["hesabin", "uyeliginiz"],
    "hesabinizi": ["hesabini", "uyeliginizi"],
    "acil": ["ivedi", "gecikmeden"],
    "sifrenizi": ["parolanizi", "sifreni"],
    "dogrulayin": ["onaylayin", "teyit edin"],
    "kacirma": ["kacirmayin", "firsati kacirma"],
    "uye": ["kayitli", "abone"],
}


class Paraphraser(Protocol):
    def paraphrase(self, text: str, n: int = 3) -> list[str]: ...


def _seed_for(base_seed: int, text: str) -> int:
    # crc32 (not builtin hash()) so the seed - and thus the whole
    # augmentation pipeline - is reproducible across processes, not just
    # within one (str hashing is randomized per-process by default).
    return base_seed ^ zlib.crc32(text.encode("utf-8"))


class TemplateParaphraser:
    def __init__(self, rng_seed: int = 42) -> None:
        self._rng_seed = rng_seed

    def paraphrase(self, text: str, n: int = 3) -> list[str]:
        rng = random.Random(_seed_for(self._rng_seed, text))
        variants: list[str] = []
        attempts = 0
        while len(variants) < n and attempts < n * 10:
            attempts += 1
            force_jitter = attempts > n * 5
            candidate = self._make_variant(text, rng, force_jitter=force_jitter)
            if candidate != text and candidate not in variants:
                variants.append(candidate)
        # Guarantee exactly n distinct-from-original variants even if the
        # synonym map ran dry for this text.
        counter = 1
        while len(variants) < n:
            fallback = f"{text} ({counter})"
            if fallback not in variants:
                variants.append(fallback)
            counter += 1
        return variants[:n]

    def _make_variant(self, text: str, rng: random.Random, *, force_jitter: bool) -> str:
        words = text.split(" ")
        new_words = []
        substituted = False
        for word in words:
            stripped = word.strip(".,!?:;")
            suffix = word[len(stripped) :] if stripped else ""
            key = stripped.lower()
            if key in SYNONYM_MAP and rng.random() < 0.7:
                new_words.append(rng.choice(SYNONYM_MAP[key]) + suffix)
                substituted = True
            else:
                new_words.append(word)
        result = " ".join(new_words)
        if force_jitter or not substituted:
            result = self._jitter(result, rng)
        return result

    @staticmethod
    def _jitter(text: str, rng: random.Random) -> str:
        options = [
            lambda t: t + "!",
            lambda t: t + " simdi",
            lambda t: (t[:-1] + "!") if t.endswith(".") else t + ".",
            lambda t: (t[0].upper() + t[1:]) if t else t,
        ]
        return rng.choice(options)(text)


def augment_examples(
    seeds: list[Record], paraphraser: Paraphraser, *, n_per_seed: int = 3, source: str = SOURCE_NAME
) -> list[Record]:
    out: list[Record] = []
    for rec in seeds:
        for variant_text in paraphraser.paraphrase(rec.text, n=n_per_seed):
            out.append(
                Record(
                    text=variant_text,
                    label=rec.label,
                    source=source,
                    lang=rec.lang,
                    extra={**rec.extra, "original_text": rec.text},
                )
            )
    return out
