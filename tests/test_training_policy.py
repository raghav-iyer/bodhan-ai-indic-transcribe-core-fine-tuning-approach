"""A tiny CPU network checks optimization policy, not Bodhan ASR quality/integration."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from marathi_asr.model import configure_trainable, keep_frozen_modules_in_eval


def test_unavailable_requested_cuda_fails_before_loading_nemo(monkeypatch):
    from marathi_asr.train import train
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        train({"training": {"accelerator": "cuda"}})


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.5))
        self.transf_decoder = nn.Module()
        self.transf_decoder.embedding = nn.Embedding(8, 4)
        self.transf_decoder.decoder = nn.Module()
        self.transf_decoder.decoder.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.transf_decoder.decoder.final_layer_norm = nn.LayerNorm(4)
        self.log_softmax = nn.Linear(4, 8, bias=False)
        self.log_softmax.weight = self.transf_decoder.embedding.weight

    def forward(self, x):
        x = self.encoder(x)
        for layer in self.transf_decoder.decoder.layers:
            x = torch.tanh(layer(x))
        return self.log_softmax(self.transf_decoder.decoder.final_layer_norm(x))


def test_top_layers_update_but_encoder_bn_and_tied_embeddings_do_not():
    torch.manual_seed(42)
    model = TinyModel()
    result = configure_trainable(model, "decoder_top", 1)
    assert 0 < result["fraction_trainable"] < 1
    initial = {n: p.detach().clone() for n, p in model.state_dict().items()}
    model.train()
    keep_frozen_modules_in_eval(model)
    assert not model.encoder.training
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    loss = nn.CrossEntropyLoss()(model(torch.randn(3, 4)), torch.tensor([0, 1, 2]))
    loss.backward()
    optimizer.step()
    after = model.state_dict()
    assert torch.equal(initial["encoder.1.running_mean"], after["encoder.1.running_mean"])
    assert torch.equal(initial["transf_decoder.embedding.weight"], after["transf_decoder.embedding.weight"])
    assert torch.equal(initial["transf_decoder.decoder.layers.0.weight"], after["transf_decoder.decoder.layers.0.weight"])
    assert not torch.equal(initial["transf_decoder.decoder.layers.2.weight"], after["transf_decoder.decoder.layers.2.weight"])


def test_wrong_layer_count_fails():
    with pytest.raises(ValueError, match="decoder layers"):
        configure_trainable(TinyModel(), "decoder_top", 10)


@pytest.mark.parametrize("final,previous,expected", [
    (0.2, None, (True, 0.2)),
    (0.2, 0.3, (True, 0.2)),
    (0.3, 0.2, (False, 0.2)),
    (0.2, 0.2, (False, 0.2)),
])
def test_endpoint_is_selected_only_when_its_validation_wer_is_better(final, previous, expected):
    from marathi_asr.train import final_checkpoint_is_better
    assert final_checkpoint_is_better([{"val_wer": final}], previous) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1])
def test_nonfinite_or_negative_endpoint_wer_cannot_be_exported(value):
    from marathi_asr.train import final_checkpoint_is_better
    with pytest.raises(FloatingPointError):
        final_checkpoint_is_better([{"val_wer": value}], 0.2)


def test_endpoint_requires_the_monitored_validation_metric():
    from marathi_asr.train import final_checkpoint_is_better
    with pytest.raises(ValueError, match="val_wer"):
        final_checkpoint_is_better([{"val_loss": 0.2}], None)


@pytest.mark.parametrize("mode", ["decoder", "full"])
def test_other_tuning_modes(mode):
    model = TinyModel()
    result = configure_trainable(model, mode)
    assert model.transf_decoder.embedding.weight.requires_grad
    assert model.encoder[0].weight.requires_grad == (mode == "full")
    if mode == "full":
        assert result["fraction_trainable"] == 1
