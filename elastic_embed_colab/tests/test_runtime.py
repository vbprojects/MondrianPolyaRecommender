import torch
import pytest

from elastic_embed.runtime import compare_gradients, record_error


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_reference_and_mohawk_do_not_invoke_torchscript_cuda_fusion(monkeypatch, device):
    import elastic_embed.model as model_module
    from elastic_embed.mohawk import mixer_coefficients

    def unavailable_nvrtc(*args):
        raise RuntimeError("nvrtc: failed to open libnvrtc-builtins.so.13.0")

    monkeypatch.setattr(model_module, "_scripted_scan", unavailable_nvrtc)
    mixer = model_module.ReferenceKDA(width=16, heads=2, head_dim=8).to(device)
    x = torch.randn(2, 7, 16, device=device, requires_grad=True)
    mask = torch.tensor([[1]*7, [1]*4+[0]*3], device=device)
    # Repeat to cover the profiling/warmup pattern that triggered JIT fusion.
    for _ in range(3):
        output = mixer(x, attention_mask=mask)[0]
        gradient = torch.autograd.grad(output.square().sum(), x)[0]
        assert torch.isfinite(output).all() and torch.isfinite(gradient).all()
    matrix = mixer_coefficients(mixer, x, mask)
    gradient = torch.autograd.grad(matrix.square().sum(), x)[0]
    assert torch.isfinite(matrix).all() and torch.isfinite(gradient).all()


def test_gradient_comparison_matches_names_and_retains_all_failures():
    reference = {"q_proj.weight": torch.ones(3, 3), "q_conv1d.weight": torch.ones(3, 1, 4)}
    reordered = dict(reversed(list(reference.items())))
    assert all(check["passed"] for check in compare_gradients(reference, reordered))
    bad = {name: value * 2 for name, value in reordered.items()}
    checks = compare_gradients(reference, bad)
    assert len(checks) == 2
    assert all(not check["passed"] for check in checks)
    assert all(check["relative_l2_error"] == 1 for check in checks)


def test_gradient_comparison_rejects_nonfinite_or_missing_parameters():
    checks = compare_gradients({"x": torch.ones(1)}, {"x": torch.tensor([float("nan")])})
    assert not checks[0]["passed"]
    assert checks[0]["relative_l2_error"] is None
    with pytest.raises(AssertionError, match="names differ"):
        compare_gradients({"x": torch.ones(1)}, {"y": torch.ones(1)})


def test_runtime_error_is_saved_and_visible(capsys):
    report = {}
    try:
        raise RuntimeError("diagnostic sentinel")
    except RuntimeError:
        record_error(report, "fla")
    assert "diagnostic sentinel" in report["fla_error"]
    assert "diagnostic sentinel" in capsys.readouterr().err


def test_bf16_cancellation_is_reported_without_false_elementwise_failure():
    expected = {"weight": torch.tensor([10., .01])}
    actual = {"weight": torch.tensor([10., .07])}
    check = compare_gradients(expected, actual, dtype=torch.bfloat16)[0]
    assert check["passed"]
    assert check["mismatched_elements"] == 1
    assert not compare_gradients(expected, actual, dtype=torch.float32)[0]["passed"]


def test_bf16_gate_rejects_drift_and_isolated_spikes():
    expected = {"weight": torch.ones(10000)}
    drift = {"weight": expected["weight"] * 1.04}
    assert not compare_gradients(expected, drift, dtype=torch.bfloat16)[0]["passed"]
    spike = {"weight": expected["weight"].clone()}
    spike["weight"][0] += .5
    check = compare_gradients(expected, spike, dtype=torch.bfloat16)[0]
    assert check["relative_l2_error"] < .03
    assert not check["passed"]
