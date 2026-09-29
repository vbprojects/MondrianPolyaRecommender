"""Download the pinned encoder to HF_HOME and verify actual CPU inference."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
model = 'TaylorAI/gte-tiny@4cc5e73d86a67c601897257b467187234aa3bca3'
worker = subprocess.Popen(
    [sys.executable, "-u", str(root / "scripts/live_embedding_worker.py"), "--device", "cpu", "--model", model],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
)
try:
    ready = json.loads(worker.stdout.readline())
    assert ready["ready"] and ready["dimension"] == 384 and ready["model"] == model, ready
    worker.stdin.write(json.dumps({"op": "embed", "texts": ["A public index of conversations.", "Trees and probability."]}) + "\n")
    worker.stdin.flush()
    result = json.loads(worker.stdout.readline())
    if "error" in result:
        raise RuntimeError(result["error"])
    assert result["provider"] == "CPUExecutionProvider", result["provider"]
    assert result["model"] == ready["model"] and result["count"] == 2
    import numpy as np
    vectors = np.frombuffer(base64.b64decode(result["vectors"]), dtype="<f4").reshape(2, 384)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    worker.stdin.write('{"op":"close"}\n')
    worker.stdin.flush()
    assert json.loads(worker.stdout.readline())["ok"]
    assert worker.wait(timeout=30) == 0
    print(json.dumps({"model": ready["model"], "dimension": 384, "provider": result["provider"],
                      "cache": os.environ.get("HF_HOME", "default"), "inference_passed": True}))
finally:
    if worker.poll() is None:
        worker.kill()
        worker.wait()
