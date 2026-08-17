import json

from spamdet.generate_synthetic import main


def test_main_writes_seeds_augmented_and_adversarial_jsonl(tmp_path):
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    (seed_dir / "spam.yaml").write_text(
        "category: spam\nexamples:\n  - text: bonus kazandiniz hemen tiklayin\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "generated"

    exit_code = main(
        ["--seed-dir", str(seed_dir), "--out-dir", str(out_dir), "--n-per-seed", "2", "--rng-seed", "1"]
    )
    assert exit_code == 0

    seeds = [json.loads(line) for line in (out_dir / "seeds.jsonl").read_text(encoding="utf-8").splitlines()]
    augmented = [json.loads(line) for line in (out_dir / "augmented.jsonl").read_text(encoding="utf-8").splitlines()]
    adversarial = [
        json.loads(line) for line in (out_dir / "adversarial.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(seeds) == 1
    assert len(augmented) == 2
    assert len(adversarial) == (1 + 2) * 2  # homoglyph + zero_width per seed+augmented record
    assert all(r["label"] == "spam" for r in seeds + augmented + adversarial)


def test_main_returns_error_when_seed_dir_empty(tmp_path):
    seed_dir = tmp_path / "empty_seeds"
    seed_dir.mkdir()
    exit_code = main(["--seed-dir", str(seed_dir), "--out-dir", str(tmp_path / "out")])
    assert exit_code == 1
