import pandas as pd
import pytest

from spamdet.loaders.base import ColumnNotFoundError, find_column, read_csv_robust


def test_find_column_matches_exact_name():
    df = pd.DataFrame({"Message": ["a"], "Group": ["spam"]})
    assert find_column(df, ["Message", "text"], dataset_name="x") == "Message"


def test_find_column_matches_case_and_whitespace_insensitively():
    df = pd.DataFrame({" message ": ["a"]})
    assert find_column(df, ["Message"], dataset_name="x") == " message "


def test_find_column_matches_second_candidate_when_first_absent():
    df = pd.DataFrame({"sms": ["a"]})
    assert find_column(df, ["Message", "sms"], dataset_name="x") == "sms"


def test_find_column_raises_and_lists_actual_columns():
    df = pd.DataFrame({"foo": ["a"], "bar": ["b"]})
    with pytest.raises(ColumnNotFoundError) as excinfo:
        find_column(df, ["Message", "text"], dataset_name="turkish_sms_collection")
    assert "foo" in str(excinfo.value)
    assert "bar" in str(excinfo.value)
    assert "turkish_sms_collection" in str(excinfo.value)


def test_read_csv_robust_reads_utf8(tmp_path):
    path = tmp_path / "utf8.csv"
    path.write_text("text,label\nÇok teşekkürler,legitimate\n", encoding="utf-8")
    df = read_csv_robust(path)
    assert df.loc[0, "text"] == "Çok teşekkürler"


def test_read_csv_robust_falls_back_to_cp1254(tmp_path):
    path = tmp_path / "cp1254.csv"
    content = "text,label\nŞifrenizi güncelleyin,phishing\n"
    path.write_bytes(content.encode("cp1254"))
    df = read_csv_robust(path)
    assert df.loc[0, "text"] == "Şifrenizi güncelleyin"


def test_read_csv_robust_falls_back_to_utf8_sig(tmp_path):
    path = tmp_path / "bom.csv"
    content = "text,label\nBonus kazandınız,gambling_scam\n"
    path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
    df = read_csv_robust(path)
    assert list(df.columns) == ["text", "label"]
