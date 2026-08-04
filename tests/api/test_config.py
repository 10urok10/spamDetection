from spamdet.api import config


def test_get_model_dir_uses_env_var_when_set(monkeypatch):
    monkeypatch.setenv("SPAMDET_MODEL_DIR", "/some/custom/path")
    assert config.get_model_dir() == config.Path("/some/custom/path")


def test_get_model_dir_falls_back_to_onnx_or_pytorch_default(monkeypatch):
    monkeypatch.delenv("SPAMDET_MODEL_DIR", raising=False)
    result = config.get_model_dir()
    assert result in (config.DEFAULT_ONNX_MODEL_DIR, config.DEFAULT_PYTORCH_MODEL_DIR)


def test_get_redis_url_default(monkeypatch):
    monkeypatch.delenv("SPAMDET_REDIS_URL", raising=False)
    assert config.get_redis_url() == "redis://localhost:6379/0"


def test_get_redis_url_env_override(monkeypatch):
    monkeypatch.setenv("SPAMDET_REDIS_URL", "redis://example:1234/1")
    assert config.get_redis_url() == "redis://example:1234/1"


def test_get_confirmed_data_path_default(monkeypatch):
    monkeypatch.delenv("SPAMDET_CONFIRMED_DATA_PATH", raising=False)
    assert config.get_confirmed_data_path() == config.PROJECT_ROOT / "data" / "review" / "confirmed.jsonl"


def test_get_confirmed_data_path_env_override(monkeypatch, tmp_path):
    custom = tmp_path / "custom.jsonl"
    monkeypatch.setenv("SPAMDET_CONFIRMED_DATA_PATH", str(custom))
    assert config.get_confirmed_data_path() == custom
