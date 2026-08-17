import warnings

import pandas as pd
import pytest

from spamdet.model.dataset import build_training_dataframe, read_split, split_dataset, write_split
from spamdet.schema import Label, Lang, Record


def _write_seed_file(seed_dir, name, category, n=4):
    lines = [f"category: {category}", "examples:"]
    for i in range(n):
        lines.append(f'  - text: "{category} ornek metin {i}"')
    (seed_dir / f"{name}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_build_training_dataframe_combines_synthetic_sources_offline(tmp_path):
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    _write_seed_file(seed_dir, "spam", "spam")
    _write_seed_file(seed_dir, "info", "bilgilendirme")

    df = build_training_dataframe(
        raw_dir=tmp_path / "raw",  # empty -> all public loaders skipped
        seed_dir=seed_dir,
        include_turkishsms_ds=False,
        n_per_seed=2,
    )
    assert not df.empty
    assert set(df["source"].unique()) <= {"synthetic_seed", "synthetic_augmented", "synthetic_adversarial"}
    assert set(df["label"].unique()) == {"spam", "bilgilendirme"}


def _make_balanced_df(n_per_class=20):
    records = []
    for label in (Label.BILGILENDIRME, Label.SPAM, Label.REKLAM):
        for i in range(n_per_class):
            records.append(Record(text=f"{label.value} text {i}", label=label, source="x", lang=Lang.TR))
    from spamdet.merge import records_to_dataframe

    return records_to_dataframe(records)


def test_split_dataset_is_stratified_and_covers_all_rows():
    df = _make_balanced_df(n_per_class=20)
    train, val, test = split_dataset(df, val_size=0.2, test_size=0.2, random_state=1)
    assert len(train) + len(val) + len(test) == len(df)
    for split_df in (train, val, test):
        assert set(split_df["label"].unique()) == {"bilgilendirme", "spam", "reklam"}


def test_split_dataset_falls_back_when_class_too_small_to_stratify():
    df = _make_balanced_df(n_per_class=20)
    rare_row = df.iloc[[0]].copy()
    rare_row["label"] = "otp"
    df = pd.concat([df, rare_row], ignore_index=True)  # exactly 1 member - can't stratify

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train, val, test = split_dataset(df, val_size=0.2, test_size=0.2, random_state=1)
    assert any("stratified split failed" in str(w.message) for w in caught)
    assert len(train) + len(val) + len(test) == len(df)


def test_write_split_and_read_split_round_trip(tmp_path):
    df = _make_balanced_df(n_per_class=3)
    path = tmp_path / "train.parquet"
    write_split(df, path)
    reloaded = read_split(path)
    assert len(reloaded) == len(df)
    assert reloaded["extra"].iloc[0] == {}
    assert list(reloaded.columns) == list(df.columns)
