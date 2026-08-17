from spamdet.manual_labels import (
    SKIP_LABEL,
    already_decided_texts,
    append_decision,
    build_candidate_pool,
    load_manual_labels,
)
from spamdet.schema import Label


def test_append_decision_then_already_decided_texts_round_trip(tmp_path):
    path = tmp_path / "relabeled.jsonl"
    append_decision(path, text="Kacirma! Firsat seni bekliyor.", label="reklam", original_label="ham", original_source="turkish_sms_collection")
    append_decision(path, text="Kodunuz 482913, kimseyle paylasmayin.", label="otp", original_label="ham", original_source="turkish_sms_collection")

    decided = already_decided_texts(path)
    assert decided == {"Kacirma! Firsat seni bekliyor.", "Kodunuz 482913, kimseyle paylasmayin."}


def test_already_decided_texts_empty_when_file_missing(tmp_path):
    assert already_decided_texts(tmp_path / "does_not_exist.jsonl") == set()


def test_load_manual_labels_returns_records_with_correct_label_and_source(tmp_path):
    path = tmp_path / "relabeled.jsonl"
    append_decision(path, text="Kacirma! Firsat seni bekliyor.", label="reklam", original_label="ham", original_source="turkish_sms_collection")

    records = load_manual_labels(path)
    assert len(records) == 1
    assert records[0].text == "Kacirma! Firsat seni bekliyor."
    assert records[0].label is Label.REKLAM
    assert records[0].source == "manual_relabel"
    assert records[0].extra["original_label"] == "ham"
    assert records[0].extra["original_source"] == "turkish_sms_collection"


def test_load_manual_labels_excludes_skipped_entries(tmp_path):
    path = tmp_path / "relabeled.jsonl"
    append_decision(path, text="belirsiz mesaj", label=SKIP_LABEL, original_label="ham", original_source="turkish_spam_dataset")
    append_decision(path, text="net bir bilgilendirme mesaji", label="bilgilendirme", original_label="ham", original_source="turkish_spam_dataset")

    records = load_manual_labels(path)
    assert len(records) == 1
    assert records[0].text == "net bir bilgilendirme mesaji"


def test_load_manual_labels_empty_when_file_missing(tmp_path):
    assert load_manual_labels(tmp_path / "does_not_exist.jsonl") == []


def test_build_candidate_pool_only_includes_ham_rows_ordered_sms_first(tmp_path):
    raw_dir = tmp_path / "raw"
    (raw_dir / "turkish_sms_collection").mkdir(parents=True)
    (raw_dir / "turkish_sms_collection" / "sms.csv").write_text(
        "Message,Group\n"
        "Bedava bonus kazandiniz hemen tiklayin,spam\n"
        "Yarin saat 10da toplanti var,normal\n"
        "Kargonuz dagitima cikmistir,normal\n",
        encoding="utf-8",
    )
    pool = build_candidate_pool(raw_dir)

    assert len(pool) == 2  # only the two "normal"/ham rows, not the spam-labeled one
    assert all(item["original_label"] == "ham" for item in pool)
    assert all(item["original_source"] == "turkish_sms_collection" for item in pool)
    assert {item["text"] for item in pool} == {"Yarin saat 10da toplanti var", "Kargonuz dagitima cikmistir"}
