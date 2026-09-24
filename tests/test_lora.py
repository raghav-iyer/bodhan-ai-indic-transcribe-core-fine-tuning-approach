"""Verify adapter mathematics, frozen weights, export, and native NeMo targets."""

import copy
import json

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from marathi_asr.common import sha256
from marathi_asr.lora import LoRALinear, adapter_modules, load_adapters, merge_decoder_lora, save_adapters
from marathi_asr.model import configure_trainable, keep_frozen_modules_in_eval


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("query_net", "key_net", "value_net", "out_projection"):
            setattr(self, name, nn.Linear(4, 4))
        self.attn_dropout = nn.Dropout(0.4)

    def forward(self, x):
        context = torch.tanh(self.query_net(x) + self.key_net(x) + self.value_net(x))
        return self.out_projection(self.attn_dropout(context))


class TinyDecoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyAttention()  # Same leaf names must not accidentally receive adapters.
        self.transf_decoder = nn.Module()
        self.transf_decoder.embedding = nn.Embedding(8, 4)
        self.transf_decoder.decoder = nn.Module()
        self.transf_decoder.decoder.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            layer.first_sub_layer = TinyAttention()
            layer.second_sub_layer = TinyAttention()
            layer.third_sub_layer = nn.Linear(4, 4)
            self.transf_decoder.decoder.layers.append(layer)
        self.transf_decoder.decoder.final_layer_norm = nn.LayerNorm(4)
        self.log_softmax = nn.Linear(4, 8, bias=False)
        self.log_softmax.weight = self.transf_decoder.embedding.weight

    def forward(self, x):
        x = self.encoder(x)
        for layer in self.transf_decoder.decoder.layers:
            x = x + layer.first_sub_layer(x)
            x = x + layer.second_sub_layer(x)
            x = x + torch.tanh(layer.third_sub_layer(x))
        return self.log_softmax(self.transf_decoder.decoder.final_layer_norm(x))


def test_initial_lora_is_exactly_the_original_linear():
    torch.manual_seed(12)
    linear = nn.Linear(7, 5)
    inputs = torch.randn(2, 3, 7)
    expected = linear(inputs).detach()
    wrapped = LoRALinear(linear, rank=2, alpha=4, dropout=0.5)
    assert torch.equal(wrapped(inputs), expected)
    wrapped.eval()
    assert torch.equal(wrapped(inputs), expected)
    assert not wrapped.base.weight.requires_grad
    assert not wrapped.base.bias.requires_grad
    wrapped(inputs).square().mean().backward()
    assert torch.count_nonzero(wrapped.lora_A.grad) == 0
    assert torch.count_nonzero(wrapped.lora_B.grad) > 0


def test_only_adapters_update_and_frozen_decoder_dropout_stays_off():
    torch.manual_seed(42)
    model = TinyDecoderModel()
    policy = configure_trainable(model, "decoder_lora", lora={"rank": 2, "alpha": 4, "dropout": 0.1})
    assert len(policy["lora"]["target_modules"]) == 2 * 2 * 4
    assert all(name.endswith((".lora_A", ".lora_B")) for name in policy["trainable_names"])
    assert all(name.startswith("transf_decoder.decoder.layers.") for name in policy["lora"]["target_modules"])
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    model.train()
    keep_frozen_modules_in_eval(model)
    assert not model.encoder.training
    assert not model.transf_decoder.decoder.layers[0].first_sub_layer.attn_dropout.training
    assert all(module.dropout.training for module in adapter_modules(model).values())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    x, labels = torch.randn(3, 4), torch.tensor([0, 2, 1])
    for _ in range(2):
        optimizer.zero_grad()
        nn.CrossEntropyLoss()(model(x), labels).backward()
        assert all(p.grad is None for name, p in model.named_parameters()
                   if not name.endswith((".lora_A", ".lora_B")))
        assert any(module.lora_B.grad.abs().max() > 0 for module in adapter_modules(model).values())
        optimizer.step()
    for name, value in model.state_dict().items():
        if not name.endswith((".lora_A", ".lora_B")):
            assert torch.equal(initial[name], value), name
    assert any(not torch.equal(initial[name], value) for name, value in model.state_dict().items()
               if name.endswith(".lora_A"))
    assert any(not torch.equal(initial[name], value) for name, value in model.state_dict().items()
               if name.endswith(".lora_B"))


def test_merged_model_has_native_keys_and_same_predictions():
    torch.manual_seed(8)
    model = TinyDecoderModel().eval()
    native_copy = copy.deepcopy(model)
    configure_trainable(model, "decoder_lora", lora={"rank": 2, "alpha": 4, "dropout": 0.1})
    with torch.no_grad():
        for module in adapter_modules(model).values():
            module.lora_B.normal_(std=0.01)
    model.eval()
    inputs = torch.randn(3, 4)
    expected = model(inputs).detach()
    targets = merge_decoder_lora(model)
    assert len(targets) == 16
    assert not adapter_modules(model)
    assert set(model.state_dict()) == set(native_copy.state_dict())
    native_copy.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(model(inputs), expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(native_copy(inputs), expected, atol=1e-6, rtol=1e-5)


def test_adapter_export_identifies_the_exact_base_and_contains_only_adapters(tmp_path):
    base = tmp_path / "base.nemo"
    base.write_bytes(b"an exact checkpoint identity")
    model = TinyDecoderModel()
    configure_trainable(model, "decoder_lora", lora={"rank": 2})
    metadata = save_adapters(model, tmp_path / "export", base, sha256(base))
    saved = json.loads((tmp_path / "export/adapters.json").read_text())
    assert saved == metadata
    assert saved["base_checkpoint_sha256"] == sha256(base)
    assert saved["weights_sha256"] == sha256(tmp_path / "export/adapters.pt")
    tensors = torch.load(tmp_path / "export/adapters.pt", weights_only=True)
    assert len(tensors) == 32
    assert all(name.endswith((".lora_A", ".lora_B")) for name in tensors)
    for name, module in adapter_modules(model).items():
        assert saved["targets"][name]["rank"] == 2
        torch.testing.assert_close(tensors[f"{name}.lora_A"], module.lora_A)
        torch.testing.assert_close(tensors[f"{name}.lora_B"], module.lora_B)


def test_saved_adapters_reload_only_on_their_exact_base(tmp_path):
    base = tmp_path / "base.nemo"
    base.write_bytes(b"the original model")
    model = TinyDecoderModel().eval()
    native = copy.deepcopy(model)
    configure_trainable(model, "decoder_lora", lora={"rank": 2})
    with torch.no_grad():
        for module in adapter_modules(model).values():
            module.lora_B.normal_(std=0.01)
    model.eval()
    save_adapters(model, tmp_path / "export", base, sha256(base))
    wrong_base = tmp_path / "different_base.nemo"
    wrong_base.write_bytes(b"a different trained checkpoint")
    with pytest.raises(ValueError, match="base checkpoint SHA256 mismatch"):
        load_adapters(native, tmp_path / "export", wrong_base)
    assert not adapter_modules(native)
    load_adapters(native, tmp_path / "export", base)
    inputs = torch.randn(3, 4)
    torch.testing.assert_close(native(inputs), model(inputs), atol=0, rtol=0)
    assert all(name.endswith((".lora_A", ".lora_B"))
               for name, parameter in native.named_parameters() if parameter.requires_grad)


@pytest.mark.parametrize("corruption", ["hash", "target", "shape", "settings"])
def test_corrupt_adapter_artifacts_do_not_modify_the_model(tmp_path, corruption):
    base = tmp_path / "base.nemo"
    base.write_bytes(b"base")
    model = TinyDecoderModel()
    native = copy.deepcopy(model)
    configure_trainable(model, "decoder_lora", lora={"rank": 2})
    metadata = save_adapters(model, tmp_path, base, sha256(base))
    target = next(iter(metadata["targets"]))
    if corruption == "hash":
        metadata["weights_sha256"] = "not the hash"
    elif corruption == "target":
        del metadata["targets"][target]
    elif corruption == "settings":
        metadata["targets"][target]["dropout"] = 2.0
    else:
        tensors = torch.load(tmp_path / "adapters.pt", weights_only=True)
        tensors[f"{target}.lora_A"] = torch.zeros(1, 1)
        torch.save(tensors, tmp_path / "adapters.pt")
        metadata["weights_sha256"] = sha256(tmp_path / "adapters.pt")
    (tmp_path / "adapters.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_adapters(native, tmp_path, base)
    assert not adapter_modules(native)


def test_wrong_architecture_is_rejected_before_installing_any_adapters():
    model = TinyDecoderModel()
    model.transf_decoder.decoder.layers[-1].second_sub_layer.value_net = nn.Identity()
    with pytest.raises(ValueError, match="Expected a native NeMo Linear"):
        configure_trainable(model, "decoder_lora")
    assert not adapter_modules(model)


@pytest.mark.parametrize("options", [{"rank": 0}, {"rank": True}, {"alpha": float("nan")}, {"dropout": 1.0}])
def test_invalid_lora_options_fail(options):
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(4, 4), **options)


def test_real_nemo_decoder_forward_backward_and_merged_reload():
    nemo_decoder = pytest.importorskip("nemo.collections.asr.modules.transformer.transformer_decoders")
    torch.manual_seed(7)
    decoder = nemo_decoder.TransformerDecoder(
        num_layers=2, hidden_size=8, inner_size=16, num_attention_heads=2, pre_ln=True,
    ).eval()
    native_copy = copy.deepcopy(decoder)
    model = nn.Module()
    model.transf_decoder = nn.Module()
    model.transf_decoder.decoder = decoder
    states, encoder_states = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
    decoder_mask, encoder_mask = torch.ones(2, 3), torch.ones(2, 5)

    def run(module):
        # The native decoder returns (final hidden states, cross-attention scores).
        hidden_states, _ = module(states, decoder_mask, encoder_states, encoder_mask)
        return hidden_states

    expected = run(decoder).detach()
    configure_trainable(model, "decoder_lora", lora={"rank": 2, "alpha": 4, "dropout": 0.0})
    torch.testing.assert_close(run(decoder), expected, rtol=0, atol=0)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    (run(decoder) - torch.randn_like(expected)).square().mean().backward()
    assert any(module.lora_B.grad is not None and module.lora_B.grad.abs().max() > 0
               for module in adapter_modules(model).values())
    optimizer.step()
    adapted = run(decoder).detach()
    assert not torch.allclose(adapted, expected)
    merge_decoder_lora(model)
    native_copy.load_state_dict(decoder.state_dict(), strict=True)
    torch.testing.assert_close(run(native_copy), adapted, atol=1e-6, rtol=1e-5)
