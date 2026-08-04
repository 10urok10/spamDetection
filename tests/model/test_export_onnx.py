from unittest.mock import MagicMock, patch

from spamdet.model.export_onnx import export_to_onnx, main


def test_export_to_onnx_saves_model_tokenizer_and_label_map(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "label_map.json").write_text('{"label2id": {}, "id2label": {}}', encoding="utf-8")
    onnx_dir = tmp_path / "onnx_out"

    fake_ort_model = MagicMock()
    fake_tokenizer = MagicMock()

    with (
        patch(
            "spamdet.model.export_onnx.ORTModelForSequenceClassification.from_pretrained",
            return_value=fake_ort_model,
        ) as mock_from_pretrained,
        patch("spamdet.model.export_onnx.AutoTokenizer.from_pretrained", return_value=fake_tokenizer),
    ):
        result_dir = export_to_onnx(model_dir, onnx_dir)

    mock_from_pretrained.assert_called_once_with(model_dir, export=True)
    fake_ort_model.save_pretrained.assert_called_once_with(onnx_dir)
    fake_tokenizer.save_pretrained.assert_called_once_with(onnx_dir)
    assert result_dir == onnx_dir
    assert (onnx_dir / "label_map.json").exists()


def test_export_to_onnx_skips_label_map_copy_when_absent(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    onnx_dir = tmp_path / "onnx_out"

    with (
        patch("spamdet.model.export_onnx.ORTModelForSequenceClassification.from_pretrained", return_value=MagicMock()),
        patch("spamdet.model.export_onnx.AutoTokenizer.from_pretrained", return_value=MagicMock()),
    ):
        export_to_onnx(model_dir, onnx_dir)

    assert not (onnx_dir / "label_map.json").exists()


def test_main_returns_error_when_model_dir_missing(tmp_path):
    exit_code = main(["--model-dir", str(tmp_path / "nonexistent"), "--onnx-dir", str(tmp_path / "out")])
    assert exit_code == 1


def test_main_happy_path_calls_export(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    onnx_dir = tmp_path / "onnx_out"

    with patch("spamdet.model.export_onnx.export_to_onnx", return_value=onnx_dir) as mock_export:
        exit_code = main(["--model-dir", str(model_dir), "--onnx-dir", str(onnx_dir)])

    assert exit_code == 0
    mock_export.assert_called_once_with(model_dir, onnx_dir)
