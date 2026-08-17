"""Support for manually re-labeling public-dataset rows into the real
otp/reklam/bilgilendirme/spam taxonomy.

Why this exists: the public Turkish SMS datasets (turkish_sms_collection,
turkish_spam_dataset) only distinguish spam vs. ham, so the loaders map
every row to just Label.SPAM or Label.BILGILENDIRME (see docs/datasets.md).
A manual sample check of the "spam"-labeled bucket (~2,787 rows) found it
is overwhelmingly real advertising from identifiable brands (KIGILI,
Garanti Mortgage, Vatan, CarrefourSA, Mudo, ...), not fraud - these
datasets' original annotators evidently used "spam" to mean "any bulk/
commercial SMS", not this project's narrower spam definition. That means
the current pipeline is training thousands of real reklam messages as
spam examples directly, which is a much bigger source of the reklam-vs-
spam confusion than anything synthetic seed data alone can fix. The
"ham" bucket (~2,702 rows) is reviewed too, lower priority - it likely
still hides some real otp/reklam under the blanket bilgilendirme mapping,
just at a lower hit rate than the spam bucket.

Unlike the hand-written synthetic seeds (data/synthetic/seeds/), these
are real messages from many different real senders, which is exactly the
kind of diversity `docs/model.md`'s investigation history says actually
moves the needle. scripts/label_tool.py is the human-facing tool that
produces the JSONL this module reads; this module is the read/write layer
shared between that tool and model/dataset.py's training pipeline.

The output file lives under data/manual_labels/ (gitignored, same
rationale as data/review/ - see .gitignore: real message text, possibly
containing personal information even though the source dataset is
public, shouldn't be redistributed via this repo).
"""

import json
import random
from pathlib import Path

from .schema import Label, Lang, Record

SOURCE_NAME = "manual_relabel"

# Not a real Label value - a reviewer's explicit "skip this one, unclear
# or not useful" decision. Recorded so the item never resurfaces in the
# labeling tool, but excluded from training data.
SKIP_LABEL = "skip"

# Sources worth hand-reviewing: real Turkish text where the public
# dataset's binary spam/ham label plausibly hides a real otp/reklam/
# bilgilendirme message underneath. sms_spam_collection is deliberately
# excluded - it's English, no use for Turkish otp/reklam vocabulary;
# turkishsms_ds is excluded by default since it needs a live network call
# (see build_dataset.py's --offline convention) rather than local raw/
# files.
CANDIDATE_SOURCES = ("turkish_sms_collection", "turkish_spam_dataset")

# original dataset label -> the schema value the loaders currently map it
# to (both loaders' LABEL_MAP happen to be a clean binary spam/ham split,
# so this is exact, not an approximation).
_ORIGINAL_LABEL_TO_SCHEMA_VALUE = {"spam": Label.SPAM.value, "ham": Label.BILGILENDIRME.value}


def default_output_path(project_root: Path) -> Path:
    return project_root / "data" / "manual_labels" / "relabeled.jsonl"


def build_candidate_pool(raw_dir: Path, *, rng_seed: int = 42) -> list[dict]:
    """Real Turkish rows from CANDIDATE_SOURCES, as plain dicts
    ({"text", "original_label", "original_source"}) rather than Records -
    these aren't confirmed labels yet, just candidates for a human to
    review. Ordered spam-bucket first (the higher-value target - see the
    module docstring), ham-bucket second; within each, turkish_sms_
    collection (real SMS text) before turkish_spam_dataset (per
    docs/datasets.md this one is actually repurposed *email* ham/spam
    data, not SMS - forwarded threads, headers, signatures - noisier to
    review). Each source+label group is shuffled internally with a fixed
    seed so the labeling tool's ordering is stable across restarts
    (resuming mid-pool doesn't reshuffle what's already been seen).
    """
    from .build_dataset import DEFAULT_RAW_DIR, build_loaders
    from .merge import merge_sources

    raw_dir = raw_dir or DEFAULT_RAW_DIR
    loaders = {name: fn for name, fn in build_loaders(raw_dir, include_turkishsms_ds=False).items() if name in CANDIDATE_SOURCES}
    if not loaders:
        return []
    df = merge_sources(loaders)

    pool: list[dict] = []
    rng = random.Random(rng_seed)
    for original_label in ("spam", "ham"):
        schema_value = _ORIGINAL_LABEL_TO_SCHEMA_VALUE[original_label]
        for source in ("turkish_sms_collection", "turkish_spam_dataset"):
            subset = df[(df["label"] == schema_value) & (df["source"] == source)]
            rows = [
                {"text": row.text, "original_label": original_label, "original_source": row.source}
                for row in subset.itertuples()
            ]
            rng.shuffle(rows)
            pool.extend(rows)
    return pool


def already_decided_texts(path: str | Path) -> set[str]:
    """Texts that already have a decision recorded (label or skip) -
    used by the labeling tool to avoid re-showing them."""
    path = Path(path)
    if not path.exists():
        return set()
    decided: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            decided.add(json.loads(line)["text"])
    return decided


def append_decision(
    path: str | Path, *, text: str, label: str, original_label: str, original_source: str
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "text": text,
        "label": label,
        "original_label": original_label,
        "original_source": original_source,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manual_labels(path: str | Path, *, lang: Lang = Lang.TR) -> list[Record]:
    """Records from human-confirmed relabeling decisions, for
    model/dataset.py to merge into training data. Skip decisions are
    excluded - they're not a real label, just "reviewed, don't ask
    again".
    """
    path = Path(path)
    if not path.exists():
        return []
    records: list[Record] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["label"] == SKIP_LABEL:
                continue
            records.append(
                Record(
                    text=row["text"],
                    label=Label(row["label"]),
                    source=SOURCE_NAME,
                    lang=lang,
                    extra={"original_label": row["original_label"], "original_source": row["original_source"]},
                )
            )
    return records
