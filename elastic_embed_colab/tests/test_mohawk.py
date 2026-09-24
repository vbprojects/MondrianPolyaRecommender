from pathlib import Path
import copy
import json
import shutil
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from elastic_embed.config import identity, load_config
from elastic_embed.io import atomic_json, atomic_torch, checksum, seed_all
from elastic_embed.model import ReferenceKDA, RMSNorm, Student
from elastic_embed.mohawk import (STATE_FIELDS, AlignmentCache, cache_alignment, capture_alignment,
                                 matrix_loss, mixer_coefficients, select_calibration, set_trainable,
                                 train_alignment)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


class FakeAttention(nn.Module):
    def forward(self, hidden_states):
        n = hidden_states.shape[1]
        weights = torch.ones(n, n, device=hidden_states.device).tril()
        weights = weights / weights.sum(-1, keepdim=True)
        return (weights @ hidden_states) * .2, weights[None, None].expand(1, 2, -1, -1)


class FakeBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = RMSNorm(width), RMSNorm(width)
        self.self_attn = FakeAttention()

    def forward(self, hidden_states):
        h = hidden_states + self.self_attn(self.input_layernorm(hidden_states))[0]
        return h + .1 * F.silu(self.post_attention_layernorm(h))


class FakeTeacher(nn.Module):
    def __init__(self, width=4096):
        super().__init__()
        self.embedding = nn.Embedding(32, width)
        self.layers = nn.ModuleList([FakeBlock(width) for _ in range(3)])
        self.calls, self.fail = 0, False

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, **kwargs):
        self.calls += 1
        h = self.embedding(input_ids)
        for block in self.layers:
            h = block(hidden_states=h)
            if self.fail:
                raise RuntimeError('injected teacher failure')
        return SimpleNamespace(last_hidden_state=h)


def setup_run(root):
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/smoke.yaml')
    cfg['run_dir'] = str(root)
    cfg['model'].update(vocab_size=32, input_rank=8, width=16, layers=2, heads=4, head_dim=4,
                        ffn_width=24, backend='reference', depths=[1, 2], dims=[2, 4, 8, 12, 16])
    cfg['mohawk'].update(teacher_layers=[1, 3], examples=4, validation_examples=2, max_length=5,
                         matrix_length=3, shard_size=2, matrix_steps=2, block_steps=3,
                         effective_batch=2, block_freeze_ffn_steps=1)
    cfg['training'].update(device='cpu', precision='float32', microbatch=1, checkpoint_every=1, log_every=1)
    root.mkdir()
    rows = [dict(id=str(i), text_sha256=str(i), input_ids=[i+1, 2, 3, 4][:2+i%3],
                 split='train' if i < 4 else 'validation' if i < 6 else 'test', language='en' if i%2 else 'es') for i in range(7)]
    (root/'corpus.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    corpus = dict(id='corpus', corpus_sha256=checksum(root/'corpus.jsonl'), pad_id=0,
                  tokenizer_revision=cfg['teacher']['revision'], preprocessing='test')
    atomic_json(root/'corpus_manifest.json', corpus)
    seed_all(42)
    P = torch.linalg.qr(torch.randn(4096, 16)).Q
    atomic_torch(root/'bridge.pt', {'P':P})
    model = Student(cfg['model'])
    transferred = ['table.weight', 'input_proj.weight'] + [f'blocks.{j}.{part}.weight' for j in range(2) for part in ('gate','up','down')]
    atomic_torch(root/'initialized.pt', dict(model=model.state_dict(), model_config=cfg['model'], transferred=transferred,
                                          teacher_layers=[1,3], bridge_sha256=checksum(root/'bridge.pt')))
    return cfg, rows, corpus, P


def test_materialized_mixer_matches_reference_and_gradients():
    torch.manual_seed(9)
    mixer = ReferenceKDA(16,4,4)
    x = torch.randn(2,5,16,requires_grad=True)
    mask = torch.tensor([[1,1,1,1,1],[1,1,1,0,0]])
    matrix = mixer_coefficients(mixer,x,mask)
    v = mixer.v_proj(x) * mask.unsqueeze(-1)
    v = F.silu(mixer.v_conv1d(F.pad(v.transpose(1,2),(3,0))).transpose(1,2)).reshape(2,5,4,4)
    latent = torch.einsum('bhts,bshv->bthv',matrix,v)
    gated = mixer.o_norm(latent) * mixer.g_proj(x).reshape_as(latent).sigmoid()
    actual = mixer.o_proj(gated.flatten(-2)) * mask.unsqueeze(-1)
    expected = mixer(x,attention_mask=mask)[0]
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    ga = torch.autograd.grad(actual.square().sum(),[x,*mixer.parameters()],retain_graph=True)
    gb = torch.autograd.grad(expected.square().sum(),[x,*mixer.parameters()])
    for a,b in zip(ga,gb):
        torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-4)
    assert torch.count_nonzero(matrix.triu(1)) == 0


def test_matrix_loss_padding_and_scale():
    target = torch.ones(2,5,5).tril()
    mask = torch.tensor([[True]*5,[True,True,False,False,False]])
    actual = target[:,None].repeat(1,4,1,1)*.1
    actual[1,:,2:,:] = 999
    actual[1,:,:,2:] = 999
    assert matrix_loss(actual,target,mask).item() < 1e-12


def test_capture_uses_preblock_inputs_and_cleans_hooks():
    teacher = FakeTeacher(32)
    P = torch.randn(32,8)
    ids = torch.tensor([[1,2,3,4]])
    captured = capture_alignment(teacher,P,[1,3],ids,torch.ones_like(ids),3)
    with torch.no_grad():
        h = teacher.embedding(ids)
        expected_input = h[0] @ P
        for block in teacher.layers:
            h = block(h)
    torch.testing.assert_close(captured['inputs'][0],expected_input)
    torch.testing.assert_close(captured['outputs'][-1],h[0]@P)
    assert captured['matrix'].shape == (2,3,3)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in teacher.modules())
    teacher.fail = True
    with pytest.raises(RuntimeError,match='injected'):
        capture_alignment(teacher,P,[1,3],ids,torch.ones_like(ids),3)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in teacher.modules())


def test_cache_interruption_identity_tokens_and_no_test_leakage(tmp_path,monkeypatch):
    import elastic_embed.mohawk as mh
    cfg, rows, corpus, P = setup_run(tmp_path/'run')
    teacher = FakeTeacher()
    monkeypatch.setattr(mh,'load_teacher',lambda cfg,attn_implementation: teacher)
    original = mh.write_shard
    def interrupt(*args):
        original(*args)
        raise KeyboardInterrupt
    monkeypatch.setattr(mh,'write_shard',interrupt)
    with pytest.raises(KeyboardInterrupt):
        cache_alignment(cfg,True)
    assert teacher.calls == 2
    monkeypatch.setattr(mh,'write_shard',original)
    cache_alignment(cfg,True)
    assert teacher.calls == 6
    cache = AlignmentCache(cfg,rows,corpus)
    assert set(cache.index) == {str(i) for i in range(6)}
    batch = cache.batch(rows[:2],0,'cpu','mohawk-matrix')
    assert batch['inputs'].shape[1] == 3
    assert batch['mask'][0].tolist() == [True,True,False]
    with pytest.raises(ValueError,match='token view'):
        wrong = copy.deepcopy(rows); wrong[0]['input_ids'][0] = 9
        AlignmentCache(cfg,wrong,corpus)
    changed = copy.deepcopy(cfg); changed['mohawk']['teacher_layers'] = [2,3]
    with pytest.raises(ValueError,match='incompatible'):
        AlignmentCache(changed,rows,corpus)
    changed = copy.deepcopy(cfg); changed['mohawk']['max_cache_gb'] = 1e-10
    with pytest.raises(ValueError,match='exceeds'):
        cache_alignment(changed,True)
    assert {r['language'] for r in select_calibration(rows,'train',2,42)} == {'en','es'}


def test_freeze_schedule():
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/smoke.yaml')
    cfg['model'].update(vocab_size=32,input_rank=8,width=16,layers=2,heads=4,head_dim=4,ffn_width=24,backend='reference')
    model = Student(cfg['model'])
    set_trainable(model,'mohawk-matrix',0,cfg)
    assert model.blocks[0].mixer.q_proj.weight.requires_grad
    assert not model.blocks[0].mixer.v_proj.weight.requires_grad
    assert not model.blocks[0].norm1.weight.requires_grad
    set_trainable(model,'mohawk-block',0,cfg)
    assert model.blocks[0].mixer.v_proj.weight.requires_grad
    assert model.blocks[0].norm1.weight.requires_grad
    assert not model.blocks[0].up.weight.requires_grad
    set_trainable(model,'mohawk-block',cfg['mohawk']['block_freeze_ffn_steps'],cfg)
    assert model.blocks[0].up.weight.requires_grad
    assert not model.table.weight.requires_grad and not model.readout.weight.requires_grad


def test_staged_training_and_exact_resume(tmp_path,monkeypatch):
    import elastic_embed.mohawk as mh
    cfg, rows, corpus, P = setup_run(tmp_path/'baseline')
    monkeypatch.setattr(mh,'load_teacher',lambda cfg,attn_implementation: FakeTeacher())
    cache_alignment(cfg,True)
    with pytest.raises(FileNotFoundError):
        train_alignment(cfg,'mohawk-block')
    train_alignment(cfg,'mohawk-matrix')
    root = tmp_path/'baseline'
    initialized = torch.load(root/'initialized.pt',weights_only=False)
    oriented = torch.load(root/'mohawk-matrix.pt',weights_only=False)
    for name in ('table.weight','readout.weight','blocks.0.up.weight','blocks.0.mixer.v_proj.weight'):
        torch.testing.assert_close(initialized['model'][name],oriented['model'][name],atol=0,rtol=0)
    assert not torch.equal(initialized['model']['blocks.0.mixer.q_proj.weight'],oriented['model']['blocks.0.mixer.q_proj.weight'])
    interrupted = copy.deepcopy(cfg); interrupted['run_dir'] = str(tmp_path/'interrupted')
    shutil.copytree(root,tmp_path/'interrupted')
    train_alignment(cfg,'mohawk-block')
    original = mh.atomic_torch
    def stop_after_save(path,state):
        original(path,state)
        if state['step'] == 1:
            raise KeyboardInterrupt
    monkeypatch.setattr(mh,'atomic_torch',stop_after_save)
    with pytest.raises(KeyboardInterrupt):
        train_alignment(interrupted,'mohawk-block')
    first = torch.load(tmp_path/'interrupted/mohawk-block.pt',weights_only=False)
    torch.testing.assert_close(first['model']['blocks.0.up.weight'],oriented['model']['blocks.0.up.weight'],atol=0,rtol=0)
    monkeypatch.setattr(mh,'atomic_torch',original)
    train_alignment(interrupted,'mohawk-block',True)
    a = torch.load(root/'mohawk-block.pt',weights_only=False)
    b = torch.load(tmp_path/'interrupted/mohawk-block.pt',weights_only=False)
    assert a['sampler'] == b['sampler'] and a['validation_history'] == b['validation_history']
    for name in a['model']:
        torch.testing.assert_close(a['model'][name],b['model'][name],atol=0,rtol=0)
    assert not torch.equal(a['model']['blocks.0.up.weight'],oriented['model']['blocks.0.up.weight'])
    torch.testing.assert_close(a['model']['table.weight'],initialized['model']['table.weight'],atol=0,rtol=0)
    assert a['validation_history'][-1]['coverage'] == {'es':1,'en':1}
