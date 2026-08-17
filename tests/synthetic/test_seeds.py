import pytest
from pydantic import ValidationError

from spamdet.schema import Label, Lang
from spamdet.synthetic.seeds import load_all_seeds, load_seed_file


def test_load_seed_file_parses_valid_yaml(tmp_path):
    path = tmp_path / "spam.yaml"
    path.write_text(
        "category: spam\nexamples:\n  - text: bonus kazandiniz hemen tiklayin\n    notes: test note\n",
        encoding="utf-8",
    )
    seed_file = load_seed_file(path)
    assert seed_file.category is Label.SPAM
    assert len(seed_file.examples) == 1
    assert seed_file.examples[0].notes == "test note"


def test_load_seed_file_raises_on_unknown_label(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("category: not_a_real_category\nexamples:\n  - text: x\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_seed_file(path)


def test_load_all_seeds_aggregates_across_files(tmp_path):
    (tmp_path / "a.yaml").write_text(
        "category: spam\nexamples:\n  - text: sifrenizi guncelleyin\n  - text: hesabiniz kilitlendi\n",
        encoding="utf-8",
    )
    (tmp_path / "b.yaml").write_text(
        "category: bilgilendirme\nexamples:\n  - text: yarin gorusuruz\n",
        encoding="utf-8",
    )
    records = load_all_seeds(tmp_path)
    assert len(records) == 3
    assert {r.label for r in records} == {Label.SPAM, Label.BILGILENDIRME}
    assert all(r.lang is Lang.TR for r in records)
    assert all(r.source == "synthetic_seed" for r in records)


def test_actual_repo_seed_files_load_and_include_tricky_bilgilendirme_examples():
    seed_dir = "data/synthetic/seeds"
    records = load_all_seeds(seed_dir)
    assert len(records) > 0
    by_label: dict[Label, int] = {}
    for r in records:
        by_label[r.label] = by_label.get(r.label, 0) + 1
    for label in (Label.OTP, Label.REKLAM, Label.SPAM, Label.BILGILENDIRME):
        assert by_label.get(label, 0) > 0, f"missing seed examples for {label}"

    bilgilendirme_texts = [r.text.lower() for r in records if r.label is Label.BILGILENDIRME]
    assert any("http" in t or "tl" in t for t in bilgilendirme_texts), (
        "expected at least one bilgilendirme seed that looks spam-like "
        "(a real link or money amount) to guard against false positives"
    )
