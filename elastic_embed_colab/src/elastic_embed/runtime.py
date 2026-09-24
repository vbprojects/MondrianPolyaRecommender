import importlib.metadata
import json
import platform
import sys
import traceback
import torch
from .config import FLA_REVISION, run_dir
from .deployment import export_module, finish_trace, session_with_trace
from .io import atomic_json, seed_all
from .model import Block, ReferenceKDA, Student, make_mixer


def record_error(report, backend):
    error = traceback.format_exc()
    report[f"{backend}_error"] = error
    print(f"\n{backend.upper()} check failed:\n{error}", file=sys.stderr, flush=True)


def versions():
    output = {"python": platform.python_version(), "platform": platform.platform(), "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "bf16_native": torch.cuda.is_bf16_supported(including_emulation=False),
              "bf16_including_emulation": torch.cuda.is_bf16_supported(), "fla_expected_commit": FLA_REVISION}
    for name in ("torch", "flash-linear-attention", "triton", "onnx", "onnxruntime", "transformers",
                 "nvidia-cuda-nvrtc", "nvidia-cuda-nvrtc-cu12", "nvidia-cuda-nvrtc-cu13"):
        try:
            dist = importlib.metadata.distribution(name)
            output[name] = dist.version
            direct = dist.read_text("direct_url.json")
            if direct:
                output[name + "_source"] = json.loads(direct)
        except importlib.metadata.PackageNotFoundError:
            output[name] = "not installed"
    return output


def compare_gradients(reference, actual, dtype=torch.float32):
    """Match parameter names, not module registration order; retain every diagnostic."""
    if reference.keys() != actual.keys():
        raise AssertionError("Gradient parameter names differ between reference and FLA")
    checks = []
    for name, expected in reference.items():
        observed = actual[name]
        if expected.shape != observed.shape:
            raise AssertionError(f"Gradient shape differs for {name}: {expected.shape} vs {observed.shape}")
        a, b = expected.float(), observed.float()
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        delta = (a - b).abs()
        relative = float(delta.norm() / a.norm().clamp_min(1e-5)) if finite else None
        scaled_max = float(delta.max() / a.abs().max().clamp_min(1e-5)) if finite else None
        mismatches = int((delta > (.05 + .1 * b.abs())).sum())
        # BF16 reductions can disagree around zero through cancellation. Bound
        # both total error and the largest individual error relative to tensor
        # scale; retain the old elementwise diagnostic instead of discarding it.
        bf16 = dtype == torch.bfloat16
        passed = finite and (relative <= .03 and scaled_max <= .03 if bf16
                             else mismatches == 0 and relative <= .15)
        checks.append({"name": name, "shape": list(a.shape), "finite": finite,
                       "relative_l2_error": relative,
                       "max_abs_error": float(delta.max()) if finite else None,
                       "scaled_max_error": scaled_max,
                       "policy": "bf16_tensor_scale" if bf16 else "float32_elementwise",
                       "relative_l2_limit": .03 if bf16 else .15,
                       "scaled_max_limit": .03 if bf16 else None,
                       "mismatched_elements": mismatches, "elements": a.numel(),
                       "passed": passed})
    return checks


class OneBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, mask):
        return self.block(x, mask)


def runtime_check(cfg, quick=False):
    root = run_dir(cfg)
    print(f"Runtime report: {root / 'runtime_report.json'}", flush=True)
    seed_all(cfg["seed"])
    mc = dict(cfg["model"], backend="reference")
    report = {"versions": versions(), "reference_passed": False, "fla_passed": False, "onnx_passed": False,
              "untested": ["ARM", "WASM", "WebGPU", "packed training"], "tests": [],
              "state_dtype": "float32", "padding": "right; independent zero state per document", "quick": quick,
              "reference_scan": "eager PyTorch; TorchScript loop used only for ONNX export"}
    model = Student(mc)
    report["parameters"] = sum(p.numel() for p in model.parameters())
    del model
    block = Block(mc, 0).eval()
    batches = [1, 4] if quick else cfg["runtime"]["batches"]
    lengths = [1, 7, 65] if quick else cfg["runtime"]["lengths"]
    print("Checking reference backend...", flush=True)
    try:
        for batch in batches:
            for length in lengths:
                x = torch.randn(batch, length, mc["width"])
                mask = torch.ones(batch, length, dtype=torch.long)
                if batch > 1:
                    mask[-1, max(1, length // 2):] = 0
                with torch.no_grad():
                    actual = block(x, mask)
                    n = int(mask[-1].sum())
                    alone = block(x[-1:, :n], mask[-1:, :n])
                    torch.testing.assert_close(actual[-1:, :n], alone, atol=2e-5, rtol=2e-5)
                    torch.testing.assert_close(block(x, mask), actual)
                report["tests"].append({"backend": "reference", "batch": batch, "length": length, "passed": True})
        # Exercise extreme valid decay/update inputs as well as backward.
        extreme = ReferenceKDA(mc["width"], mc["heads"], mc["head_dim"])
        with torch.no_grad():
            extreme.dt_bias.copy_(torch.linspace(-20, 20, mc["width"]))
            extreme.b_proj.weight.mul_(100)
        x = torch.randn(1, 7, mc["width"], requires_grad=True)
        out = extreme(x)[0]
        out.square().mean().backward()
        assert torch.isfinite(out).all() and torch.isfinite(x.grad).all()
        report["reference_passed"] = True
    except Exception:
        record_error(report, "reference")
    print("Checking CUDA/FLA parity...", flush=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; FLA parity untested")
        source = report["versions"].get("flash-linear-attention_source", {})
        if source.get("vcs_info", {}).get("commit_id") != FLA_REVISION:
            raise RuntimeError("FLA source commit is not the pinned revision; reinstall with bootstrap_colab.sh")
        reference = ReferenceKDA(mc["width"], mc["heads"], mc["head_dim"]).cuda().eval()
        fast = make_mixer(dict(mc, backend="fla"), 0).cuda().eval()
        reference.load_state_dict(fast.state_dict(), strict=True)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        report["fla_input_dtype"] = str(dtype)
        for batch in batches:
            for length in lengths:
                x = torch.randn(batch, length, mc["width"], device="cuda")
                mask = torch.ones(batch, length, dtype=torch.long, device="cuda")
                if batch > 1:
                    mask[-1, max(1, length // 2):] = 0
                with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                    a, b = reference(x, attention_mask=mask)[0], fast(x, attention_mask=mask)[0]
                error = float((a.float() - b.float()).abs().max())
                torch.testing.assert_close(a.float(), b.float(), atol=cfg["runtime"]["fla_atol"], rtol=.03)
                report["tests"].append({"backend": "fla", "batch": batch, "length": length, "max_abs_error": error})
        x = torch.randn(2, 7, mc["width"], device="cuda", requires_grad=True)
        mask = torch.tensor([[1]*7, [1]*4+[0]*3], device="cuda")
        gradients = []
        for mixer in (reference, fast):
            mixer.train().zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                out = mixer(x, attention_mask=mask)[0]
                loss = out.float().square().sum()
            parameters = dict(mixer.named_parameters())
            grads = torch.autograd.grad(loss, [x, *parameters.values()])
            gradients.append(dict(zip(["input", *parameters], grads)))
        checks = compare_gradients(*gradients, dtype=dtype)
        report["gradient_checks"] = checks
        report["gradient_relative_errors"] = [check["relative_l2_error"] for check in checks]
        failures = [check["name"] for check in checks if not check["passed"]]
        if failures:
            raise AssertionError("Gradient parity failed for " + ", ".join(failures) +
                                 "; see gradient_checks in runtime_report.json")
        report["fla_passed"] = True
    except Exception:
        record_error(report, "fla")
    print("Checking ONNX export and parity...", flush=True)
    try:
        path = root / "one_block.onnx"
        export_module(OneBlock(block), (torch.randn(1, 7, mc["width"]), torch.ones(1, 7, dtype=torch.long)), path, ["hidden", "mask"], cfg["export"]["opset"])
        session = session_with_trace(path)
        for batch in batches:
            for length in lengths:
                x = torch.randn(batch, length, mc["width"])
                mask = torch.ones(batch, length, dtype=torch.long)
                if batch > 1:
                    mask[-1, max(1, length // 2):] = 0
                with torch.no_grad():
                    expected = block(x, mask)
                actual = torch.from_numpy(session.run(None, {"hidden": x.numpy(), "mask": mask.numpy()})[0])
                torch.testing.assert_close(actual, expected, atol=cfg["runtime"]["onnx_atol"], rtol=1e-4)
                report["tests"].append({"backend": "onnx", "batch": batch, "length": length,
                                        "max_abs_error": float((actual - expected).abs().max())})
        report["onnx"] = {"domain": "ai.onnx", "opset": cfg["export"]["opset"], "lowering": "Loop", "dtype": "float32", **finish_trace(session)}
        report["onnx_passed"] = True
    except Exception:
        record_error(report, "onnx")
    atomic_json(root / "runtime_report.json", report)
    print(json.dumps({k: report[k] for k in ("parameters", "reference_passed", "fla_passed", "onnx_passed")}, indent=2))
    if not report["reference_passed"] or (cfg["model"]["backend"] == "fla" and not report["fla_passed"]):
        raise RuntimeError(f"Runtime gate failed. Inspect {root / 'runtime_report.json'}")
