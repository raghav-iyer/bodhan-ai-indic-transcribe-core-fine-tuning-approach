import pytest

torch = pytest.importorskip("torch")

from marathi_asr.device import check_device, prepare_for_device


def test_cpu_is_explicitly_allowed():
    assert check_device("cpu") == "cpu"


def test_unavailable_mps_has_actionable_error(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="sandbox may hide Metal"):
        check_device("mps")


def test_metal_metric_conversion_includes_reset_defaults():
    metrics = pytest.importorskip("torchmetrics")

    class Accumulator(metrics.Metric):
        def __init__(self):
            super().__init__()
            self.add_state("loss", torch.tensor(0., dtype=torch.float64))
            self.add_state("count", torch.tensor(0, dtype=torch.int64))

        def update(self):
            self.loss += 1
            self.count += 1

        def compute(self):
            return self.loss

    metric = Accumulator()
    prepare_for_device(metric, "mps")  # dtype conversion itself needs no Metal device
    metric.update()
    metric.reset()
    assert metric.loss.dtype == torch.float32
    assert metric.count.dtype == torch.int64
