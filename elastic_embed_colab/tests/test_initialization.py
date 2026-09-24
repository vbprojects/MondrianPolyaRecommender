import sys
from types import SimpleNamespace

import pytest
import torch

from elastic_embed.initialization import TeacherTensors


@pytest.mark.parametrize("prefix", ["", "model."])
@pytest.mark.parametrize("name", ["embed_tokens.weight", "layers.2.mlp.gate_proj.weight",
                                 "layers.2.mlp.up_proj.weight", "layers.2.mlp.down_proj.weight"])
def test_teacher_tensor_uses_resolved_name_for_shard_and_tensor(monkeypatch, prefix, name):
    source = TeacherTensors.__new__(TeacherTensors)
    resolved = prefix + name
    source.index = {resolved: "weights.safetensors"}
    source.resolved_names = {}
    downloads, reads = [], []
    expected = torch.arange(6).reshape(2, 3)

    def download(filename):
        downloads.append(filename)
        return filename

    class Shard:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_tensor(self, key):
            reads.append(key)
            assert key == resolved
            return expected

    def safe_open(path, framework, device):
        assert (path, framework, device) == ("weights.safetensors", "pt", "cpu")
        return Shard()

    source.download = download
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=safe_open))
    assert source.tensor(name) is expected
    assert source.tensor("model." + name) is expected
    assert downloads == ["weights.safetensors"] * 2
    assert reads == [resolved] * 2
    assert source.resolved_names[name] == resolved


def test_missing_teacher_tensor_has_actionable_error():
    source = TeacherTensors.__new__(TeacherTensors)
    source.index = {"embed_tokens.weight": "weights.safetensors"}
    with pytest.raises(KeyError, match="Example available keys"):
        source.tensor("layers.0.mlp.gate_proj.weight")
