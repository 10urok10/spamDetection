from pathlib import Path

import yaml
from pydantic import BaseModel

from ..schema import Label, Lang, Record

SOURCE_NAME = "synthetic_seed"


class SeedExample(BaseModel):
    text: str
    notes: str | None = None
    # Only meaningful for category=legitimate: otp/bilgilendirme/reklam -
    # used to train the separate legitimate-message subtype classifier
    # (see spamdet.subtype). Ignored by Stage 1/2 training, which only
    # ever looks at the file's top-level `category`.
    subtype: str | None = None


class SeedFile(BaseModel):
    category: Label
    examples: list[SeedExample]


def load_seed_file(path: str | Path) -> SeedFile:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return SeedFile.model_validate(data)


def load_all_seeds(seed_dir: str | Path, *, lang: Lang = Lang.TR) -> list[Record]:
    """Load every *.yaml file in ``seed_dir`` and flatten them into Records,
    each example's label taken from its file's ``category``.
    """
    seed_dir = Path(seed_dir)
    records: list[Record] = []
    for path in sorted(seed_dir.glob("*.yaml")):
        seed_file = load_seed_file(path)
        for example in seed_file.examples:
            extra = {"notes": example.notes} if example.notes else {}
            records.append(
                Record(text=example.text, label=seed_file.category, source=SOURCE_NAME, lang=lang, extra=extra)
            )
    return records


def load_subtype_training_data(seed_dir: str | Path) -> list[tuple[str, str]]:
    """(text, subtype) pairs for every seed example tagged with a
    ``subtype`` - the training data for the legitimate-message subtype
    classifier (data/synthetic/seeds/legitimate.yaml's otp/bilgilendirme/
    reklam tags).
    """
    seed_dir = Path(seed_dir)
    pairs: list[tuple[str, str]] = []
    for path in sorted(seed_dir.glob("*.yaml")):
        seed_file = load_seed_file(path)
        for example in seed_file.examples:
            if example.subtype is not None:
                pairs.append((example.text, example.subtype))
    return pairs
