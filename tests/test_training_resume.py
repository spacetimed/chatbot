from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch

from chatbot import train
from chatbot.config import GPTConfig, TrainConfig
from chatbot.tokenizer import BPETokenizer


@pytest.fixture
def local_training(monkeypatch):
    tokenizer = BPETokenizer(256)
    tokens = torch.arange(25, dtype=torch.long)
    monkeypatch.setattr(train, "prepare_data", lambda config: (tokenizer, tokens, tokens))
    monkeypatch.setattr(train, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(train, "GPTConfig", lambda **kwargs: GPTConfig(**{
        "block_size": 4, "n_embed": 8, "n_head": 2, "n_layer": 1, **kwargs,
    }))
    for name in ("set_tracking_uri", "set_experiment", "log_params", "log_metrics"):
        monkeypatch.setattr(train.mlflow, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(train.mlflow, "start_run", lambda **kwargs: nullcontext())

    def run(config):
        monkeypatch.setattr(train, "TrainConfig", lambda: config)
        train.main()

    return run


def test_resume_matches_uninterrupted_training(tmp_path, monkeypatch, local_training):
    config = TrainConfig(
        checkpoint_dir=tmp_path / "continuous", batch_size=2, max_steps=6,
        warmup_steps=1, eval_interval=2, eval_batches=1, log_interval=2,
    )
    local_training(config)
    expected = torch.load(config.checkpoint_dir / "latest.pt", map_location="cpu")

    config = replace(config, checkpoint_dir=tmp_path / "resumed")
    evaluate = train.run_evaluation

    def interrupt(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        if kwargs["step"] == 4:
            raise RuntimeError("simulated interruption")
        return result

    monkeypatch.setattr(train, "run_evaluation", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        local_training(config)

    evaluated_steps = []

    def record_evaluation(*args, **kwargs):
        evaluated_steps.append(kwargs["step"])
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(train, "run_evaluation", record_evaluation)
    path = config.checkpoint_dir / "latest.pt"
    local_training(replace(config, resume_from=path))
    actual = torch.load(path, map_location="cpu")

    assert evaluated_steps == [6]
    assert actual["step"] == expected["step"] == 6
    assert actual["best_val_loss"] == expected["best_val_loss"]
    assert actual["best_val_step"] == expected["best_val_step"]
    assert actual["train_config"]["resume_from"] == str(path)
    for key, weight in expected["model_state_dict"].items():
        torch.testing.assert_close(actual["model_state_dict"][key], weight, rtol=0, atol=0)
    for key, state in expected["optimizer_state_dict"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(actual["optimizer_state_dict"]["state"][key][name], value, rtol=0, atol=0)


def test_completed_resume_preserves_checkpoint(tmp_path, local_training):
    path = tmp_path / "latest.pt"
    torch.save({"step": 6}, path)
    original = path.read_bytes()
    config = TrainConfig(checkpoint_dir=tmp_path, resume_from=path, max_steps=6)

    with pytest.raises(ValueError, match="max_steps must be greater"):
        local_training(config)

    assert path.read_bytes() == original
