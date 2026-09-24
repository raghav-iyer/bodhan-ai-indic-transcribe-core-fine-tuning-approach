import json

import pytest

from marathi_asr.common import load_config, load_local_token


def test_env_is_parsed_without_shell_execution(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    path = tmp_path / ".env"
    path.write_text('# comment\nUNRELATED=secret\nHF_TOKEN="hf_test_example"\n')
    load_local_token(path)
    import os
    assert os.environ["HF_TOKEN"] == "hf_test_example"
    assert "UNRELATED" not in os.environ
    path.write_text('HF_TOKEN=replacement\n')
    load_local_token(path)
    assert os.environ["HF_TOKEN"] == "hf_test_example"


def test_config_must_pin_revision(tmp_path):
    cfg = load_config("configs/marathi.json")
    cfg["model"]["revision"] = "main"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="immutable"):
        load_config(path)
