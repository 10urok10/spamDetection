from spamdet.build_dataset import build_loaders, main


def test_build_loaders_skips_missing_sources(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    loaders = build_loaders(raw_dir, include_turkishsms_ds=False)
    assert loaders == {}


def test_build_loaders_picks_up_present_csv(tmp_path):
    raw_dir = tmp_path / "raw"
    (raw_dir / "turkish_sms_collection").mkdir(parents=True)
    (raw_dir / "turkish_sms_collection" / "data.csv").write_text(
        "Message,Group\nbedava bonus,spam\n", encoding="utf-8"
    )
    loaders = build_loaders(raw_dir, include_turkishsms_ds=False)
    assert "turkish_sms_collection" in loaders
    records = loaders["turkish_sms_collection"]()
    assert len(records) == 1


def test_build_loaders_enron_prefers_folder_layout_then_csv_fallback(tmp_path):
    raw_dir = tmp_path / "raw"
    enron_dir = raw_dir / "enron_spam"
    (enron_dir / "spam").mkdir(parents=True)
    (enron_dir / "spam" / "1.txt").write_text("cheap meds", encoding="utf-8")
    loaders = build_loaders(raw_dir, include_turkishsms_ds=False)
    records = loaders["enron_spam"]()
    assert len(records) == 1

    # now folder layout absent, only a CSV present -> should fall back
    raw_dir2 = tmp_path / "raw2"
    enron_dir2 = raw_dir2 / "enron_spam"
    enron_dir2.mkdir(parents=True)
    (enron_dir2 / "dump.csv").write_text("text,label\ncheap meds,spam\n", encoding="utf-8")
    loaders2 = build_loaders(raw_dir2, include_turkishsms_ds=False)
    records2 = loaders2["enron_spam"]()
    assert len(records2) == 1


def test_main_returns_error_code_when_nothing_available(tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "out"
    exit_code = main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir), "--offline"])
    assert exit_code == 1


def test_main_writes_output_when_one_source_present(tmp_path):
    raw_dir = tmp_path / "raw"
    (raw_dir / "turkish_sms_collection").mkdir(parents=True)
    (raw_dir / "turkish_sms_collection" / "data.csv").write_text(
        "Message,Group\nbedava bonus,spam\nnormal mesaj,normal\n", encoding="utf-8"
    )
    out_dir = tmp_path / "out"
    exit_code = main(["--raw-dir", str(raw_dir), "--out-dir", str(out_dir), "--offline"])
    assert exit_code == 0
    assert (out_dir / "merged_dataset.csv").exists()
