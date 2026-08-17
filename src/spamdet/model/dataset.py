import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ..build_dataset import DEFAULT_RAW_DIR, build_loaders
from ..merge import merge_sources, records_to_dataframe
from ..schema import Label
from ..synthetic.adversarial import generate_adversarial_set
from ..synthetic.augment import TemplateParaphraser, augment_examples
from ..synthetic.seeds import load_all_seeds

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_SEED_DIR = PROJECT_ROOT / "data" / "synthetic" / "seeds"
DEFAULT_TRAIN_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "train"


def build_training_dataframe(
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    seed_dir: Path = DEFAULT_SEED_DIR,
    include_turkishsms_ds: bool = True,
    n_per_seed: int = 3,
    augment_rng_seed: int = 42,
) -> pd.DataFrame:
    """Combine whatever public source datasets are available under
    ``raw_dir`` with the synthetic seed/augmented/adversarial data into one
    deduplicated training table. Public sources with no raw files present
    are silently skipped (same behavior as build_dataset.main), since
    Stage 1's build already prints which sources were skipped and why.
    """
    loaders = build_loaders(raw_dir, include_turkishsms_ds=include_turkishsms_ds)
    public_df = merge_sources(loaders) if loaders else records_to_dataframe([])

    seeds = load_all_seeds(seed_dir)
    paraphraser = TemplateParaphraser(rng_seed=augment_rng_seed)
    augmented = augment_examples(seeds, paraphraser, n_per_seed=n_per_seed)
    adversarial = generate_adversarial_set(seeds + augmented, rng_seed=augment_rng_seed)
    synthetic_df = records_to_dataframe(seeds + augmented + adversarial)

    combined = pd.concat([public_df, synthetic_df], ignore_index=True)
    if combined.empty:
        return combined
    combined = combined.drop_duplicates(subset=["text", "lang"], keep="first").reset_index(drop=True)

    # otp is rule-detected (otp_rule.detect_otp), never predicted by the
    # ML model - see model/labels.py. Excluded here rather than at the
    # seed-file level so data/synthetic/seeds/otp.yaml can still be used
    # elsewhere (e.g. as otp_rule regression-test fixtures) without
    # needing a separate directory.
    combined = combined[combined["label"] != Label.OTP.value].reset_index(drop=True)
    return combined


def split_dataset(
    df: pd.DataFrame, *, val_size: float = 0.1, test_size: float = 0.1, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split by label. Falls back to a
    non-stratified split (with a warning) if any class is too small to
    stratify - this can happen with a handful of rare labels before the
    dataset grows past the MVP synthetic-only stage.
    """
    try:
        train_val, test = train_test_split(
            df, test_size=test_size, stratify=df["label"], random_state=random_state
        )
        relative_val = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=relative_val, stratify=train_val["label"], random_state=random_state
        )
    except ValueError as exc:
        warnings.warn(f"stratified split failed ({exc}); falling back to a non-stratified split")
        train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
        relative_val = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=relative_val, random_state=random_state)

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def write_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.assign(extra=df["extra"].apply(json.dumps)).to_parquet(path, index=False)


def read_split(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["extra"] = df["extra"].apply(json.loads)
    return df
