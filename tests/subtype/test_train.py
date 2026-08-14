from spamdet.subtype.ad_info_classifier import AdInfoClassifier
from spamdet.subtype.train import load_raw_pairs, main


def _write_seed_file(seed_dir, n_reklam=6, n_bilgilendirme=6, n_otp=2):
    lines = ["category: legitimate", "examples:"]
    for i in range(n_reklam):
        lines.append(f'  - text: "reklam ornek metin yuzde {i} indirim kampanya"')
        lines.append("    subtype: reklam")
    for i in range(n_bilgilendirme):
        lines.append(f'  - text: "bilgilendirme ornek metin fatura kargo {i}"')
        lines.append("    subtype: bilgilendirme")
    for i in range(n_otp):
        lines.append(f'  - text: "otp ornek metin kod {i}"')
        lines.append("    subtype: otp")
    (seed_dir / "legitimate.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_raw_pairs_excludes_otp(tmp_path):
    _write_seed_file(tmp_path, n_reklam=3, n_bilgilendirme=3, n_otp=2)
    pairs = load_raw_pairs(tmp_path)
    assert len(pairs) == 6  # otp excluded
    subtypes = {s for _, s in pairs}
    assert subtypes == {"reklam", "bilgilendirme"}


def test_main_trains_and_saves_model(tmp_path):
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    _write_seed_file(seed_dir, n_reklam=8, n_bilgilendirme=8, n_otp=2)
    model_path = tmp_path / "model.joblib"

    exit_code = main(["--seed-dir", str(seed_dir), "--model-path", str(model_path), "--n-per-seed", "2"])

    assert exit_code == 0
    assert model_path.exists()
    assert model_path.with_suffix(".metrics.json").exists()
    loaded = AdInfoClassifier.load(model_path)
    assert loaded.predict("reklam ornek metin yuzde 1 indirim kampanya").subtype in ("reklam", "bilgilendirme")


def test_main_returns_error_when_no_subtype_data(tmp_path):
    seed_dir = tmp_path / "empty_seeds"
    seed_dir.mkdir()
    exit_code = main(["--seed-dir", str(seed_dir), "--model-path", str(tmp_path / "model.joblib")])
    assert exit_code == 1
