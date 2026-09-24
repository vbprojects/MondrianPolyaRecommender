"""FLA-compatible parameter layout; reference math is deliberately slow.

Reference backend implements the pinned FLA KDA configuration: no GVA,
positive beta, unbounded softplus decay, causal short convolutions. State is
FP32 and [batch, head, key, value]; no state persists between calls.
"""
import math
from typing import List
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


def _scan_impl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor,
         beta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    state = torch.zeros((q.size(0), q.size(2), q.size(3), v.size(3)), device=q.device, dtype=torch.float32)
    outputs = torch.jit.annotate(List[torch.Tensor], [])
    for t in range(q.size(1)):
        decayed = state * decay[:, t].unsqueeze(-1)
        error = v[:, t] - (decayed * k[:, t].unsqueeze(-1)).sum(-2)
        updated = decayed + (beta[:, t].unsqueeze(-1) * k[:, t]).unsqueeze(-1) * error.unsqueeze(-2)
        valid = mask[:, t].reshape(-1, 1, 1, 1).to(torch.bool)
        state = torch.where(valid, updated, state)
        out = (state * q[:, t].unsqueeze(-1)).sum(-2) / math.sqrt(float(q.size(3)))
        outputs.append(out * mask[:, t].reshape(-1, 1, 1))
    return torch.stack(outputs, dim=1)


# Export needs a scripted loop to preserve a dynamic sequence length. Ordinary
# execution uses the same implementation eagerly: TorchScript's CUDA fusion can
# otherwise invoke NVRTC and fail on Colab's missing/mismatched builtins library.
_scripted_scan = torch.jit.script(_scan_impl)


def scan(q, k, v, decay, beta, mask):
    if torch.onnx.is_in_onnx_export():
        return _scripted_scan(q, k, v, decay, beta, mask)
    return _scan_impl(q, k, v, decay, beta, mask)


class ReferenceKDA(nn.Module):
    def __init__(self, width=384, heads=6, head_dim=64, conv_size=4):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.conv_size = conv_size
        for name in ("q", "k", "v"):
            setattr(self, name + "_proj", nn.Linear(width, width, bias=False))
            setattr(self, name + "_conv1d", nn.Conv1d(width, width, conv_size, groups=width, bias=False))
        self.f_proj = nn.Sequential(nn.Linear(width, head_dim, bias=False), nn.Linear(head_dim, width, bias=False))
        self.b_proj = nn.Linear(width, heads, bias=False)
        self.A_log = nn.Parameter(torch.empty(heads).uniform_(1, 16).log())
        dt = torch.exp(torch.empty(width).uniform_(math.log(0.001), math.log(0.1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log._no_weight_decay = self.dt_bias._no_weight_decay = True
        self.g_proj = nn.Sequential(nn.Linear(width, head_dim, bias=False), nn.Linear(head_dim, width))
        self.o_norm = RMSNorm(head_dim)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, x, attention_mask=None, **kwargs):
        mask = attention_mask if attention_mask is not None else torch.ones_like(x[..., 0])
        qkv = []
        for name in ("q", "k", "v"):
            z = getattr(self, name + "_proj")(x) * mask.unsqueeze(-1)
            z = getattr(self, name + "_conv1d")(F.pad(z.transpose(1, 2), (self.conv_size - 1, 0)))
            qkv.append(F.silu(z.transpose(1, 2)).reshape(x.shape[0], x.shape[1], self.heads, self.head_dim))
        q, k, v = qkv
        # FLA l2_norm uses rsqrt(sum(x*x) + 1e-6), not F.normalize's clamp.
        q = q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
        k = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
        gates = self.f_proj(x).float() + self.dt_bias.float()
        decay = (-self.A_log.float().exp().view(1, 1, -1, 1) *
                 F.softplus(gates.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim))).exp()
        o = scan(q, k, v.float(), decay, self.b_proj(x).float().sigmoid(), mask)
        gate = self.g_proj(x).reshape_as(o).float().sigmoid()
        o = (self.o_norm(o) * gate).to(x.dtype).flatten(-2)
        return self.o_proj(o) * mask.unsqueeze(-1), None, None


def make_mixer(cfg, index):
    if cfg["backend"] == "reference":
        return ReferenceKDA(cfg["width"], cfg["heads"], cfg["head_dim"], cfg["conv_size"])
    from fla.layers.kda import KimiDeltaAttention
    return KimiDeltaAttention(hidden_size=cfg["width"], num_heads=cfg["heads"], head_dim=cfg["head_dim"],
                              expand_v=1, conv_size=cfg["conv_size"], conv_bias=False, layer_idx=index,
                              allow_neg_eigval=False, safe_gate=False, mode="chunk")


class Block(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        w, f = cfg["width"], cfg["ffn_width"]
        self.norm1, self.norm2 = RMSNorm(w), RMSNorm(w)
        self.mixer = make_mixer(cfg, index)
        self.gate = nn.Linear(w, f, bias=False)
        self.up = nn.Linear(w, f, bias=False)
        self.down = nn.Linear(f, w, bias=False)

    def forward(self, x, mask):
        x = x + self.mixer(self.norm1(x), attention_mask=mask)[0]
        n = self.norm2(x)
        return x + self.down(F.silu(self.gate(n)) * self.up(n))


def validate_inputs(ids, mask):
    if ids.ndim != 2 or ids.shape != mask.shape or ids.shape[1] == 0:
        raise ValueError("Expected nonempty [batch, length] input IDs and mask")
    if not bool(((mask == 0) | (mask == 1)).all()) or not bool((mask.sum(-1) > 0).all()):
        raise ValueError("Mask must be binary; all-padding inputs are rejected")
    if bool((mask[:, 1:] > mask[:, :-1]).any()):
        raise ValueError("Only right padding is supported")


def last_valid(x, mask):
    indices = (mask.long().sum(-1) - 1).view(-1, 1, 1).expand(-1, 1, x.shape[-1])
    return x.gather(1, indices).squeeze(1)


def prefix(x, dim):
    return F.normalize(x[..., :dim].float(), dim=-1)


class Student(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.table = nn.Embedding(cfg["vocab_size"], cfg["input_rank"])
        nn.init.normal_(self.table.weight, std=0.02)
        self.input_proj = nn.Linear(cfg["input_rank"], cfg["width"], bias=False)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg["layers"])])
        self.readout_norm = RMSNorm(cfg["width"])
        self.readout = nn.Linear(cfg["width"], cfg["width"], bias=False)
        count = sum(p.numel() for p in self.parameters())
        if count > 80_000_000:
            raise ValueError(f"Student has {count:,} parameters, above 80M")

    def forward(self, ids, mask, depths=None, return_hidden=False):
        depths = tuple(self.cfg["depths"] if depths is None else depths)
        if not depths or any(d not in self.cfg["depths"] for d in depths):
            raise ValueError(f"Unsupported depths: {depths}")
        if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
            validate_inputs(ids, mask)
        x = self.input_proj(self.table(ids))
        outputs, states = {}, {}
        for i, block in enumerate(self.blocks, 1):
            x = block(x, mask)
            if i in depths:
                outputs[i] = self.readout(self.readout_norm(last_valid(x, mask)))
                if return_hidden:
                    states[i] = x
            if i == max(depths):
                break  # Actual compute termination, also truncates exported graphs.
        return (outputs, states) if return_hidden else outputs

    def embed(self, ids, mask, depth=12, truncate_dim=384, normalize=True):
        if truncate_dim not in self.cfg["dims"]:
            raise ValueError(f"Unsupported dimension: {truncate_dim}")
        z = self(ids, mask, (depth,))[depth]
        return prefix(z, truncate_dim) if normalize else z[..., :truncate_dim]
