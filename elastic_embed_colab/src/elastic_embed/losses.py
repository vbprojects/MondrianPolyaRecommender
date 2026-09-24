import torch
from torch.nn import functional as F
from .model import prefix


def pair_mask(rows, device):
    mask = torch.ones(len(rows), len(rows), dtype=torch.bool, device=device)
    for i, a in enumerate(rows):
        for j, b in enumerate(rows):
            if i == j or a["id"] == b["id"] or any(a.get(k) and a.get(k) == b.get(k) for k in ("duplicate_group_id", "parallel_group_id", "split_group")):
                mask[i, j] = False
    return mask


def distillation(outputs, targets, rows, dims, cfg, shallow_ramp=1.0):
    depths = list(outputs)
    dweights = dict(zip(cfg["model"]["dims"], [0.5, 0.5, 0.75, 1, 2]))
    lweights = dict(zip(cfg["model"]["depths"], [0.25, 0.5, 1, 1, 2]))
    weights = {(l, d): lweights[l] * dweights[d] * (shallow_ramp if l < 12 else 1) for l in depths for d in dims}
    total_weight = sum(weights.values())
    mask = pair_mask(rows, targets.device)
    loss, metrics, depth_losses = targets.sum() * 0, {}, []
    for l in depths:
        for d in dims:
            s, t = prefix(outputs[l], d), prefix(targets, d)
            direct = (1 - (s * t).sum(-1)).mean()
            rel = ((s @ s.T - t @ t.T)[mask]).square().mean() if bool(mask.any()) else direct * 0
            loss = loss + weights[l, d] / total_weight * (direct + cfg["training"]["relational_weight"] * rel)
            if l < 12 and 12 in outputs:
                depth_losses.append((1 - (s * prefix(outputs[12].detach(), d)).sum(-1)).mean())
            metrics[f"{l}/{d}/direct"] = float(direct.detach())
            metrics[f"{l}/{d}/relational"] = float(rel.detach())
            metrics[f"{l}/{d}/weight"] = weights[l, d] / total_weight
    if depth_losses:
        loss = loss + cfg["training"]["depth_weight"] * shallow_ramp * torch.stack(depth_losses).mean()
    return loss, metrics


def recovery_loss(states, target, positions, valid):
    losses = []
    for j, h in enumerate(states.values()):
        index = positions.unsqueeze(-1).expand(-1, -1, h.shape[-1])
        s = F.normalize(h.gather(1, index).float(), dim=-1)
        t = F.normalize(target[:, j].float(), dim=-1)
        losses.append((s - t).square().mean(-1)[valid].mean())
    return torch.stack(losses).mean()
