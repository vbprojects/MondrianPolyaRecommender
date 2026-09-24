import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from elastic_embed.config import load_config
from elastic_embed.data import quotas, split_groups
from elastic_embed.io import atomic_json, atomic_torch, checksum, seed_all
from elastic_embed.losses import distillation
from elastic_embed.model import Student, prefix
from elastic_embed.teacher import TargetCache, cache_contract, open_cache, write_shard
from elastic_embed.training import BatchStream


def small_model():
    cfg = dict(vocab_size=32, input_rank=8, width=16, layers=12, heads=4, head_dim=4,
               conv_size=4, ffn_width=24, backend="reference", depths=[1, 3, 6, 9, 12], dims=[2, 4, 8, 12, 16])
    return Student(cfg)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_early_exit_single_forward_and_padding_isolation():
    torch.manual_seed(5)
    model = small_model().eval()
    calls = [0] * 12
    def count(i):
        def hook(*args):
            calls[i] += 1
        return hook
    hooks = [b.register_forward_hook(count(i)) for i, b in enumerate(model.blocks)]
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    full = model(ids, mask)
    assert calls == [1] * 12
    calls[:] = [0] * 12
    early = model.embed(ids, mask, 3, 4)
    assert calls == [1, 1, 1] + [0] * 9
    torch.testing.assert_close(early, prefix(full[3], 4))
    alone = model.embed(ids[1:, :2], mask[1:, :2], 3, 4)
    torch.testing.assert_close(early[1:], alone)
    changed = ids.clone(); changed[1, 2:] = 17
    torch.testing.assert_close(early, model.embed(changed, mask, 3, 4))
    for hook in hooks:
        hook.remove()


def test_invalid_inputs_and_profiles():
    model = small_model()
    ids = torch.ones(1, 3, dtype=torch.long)
    for mask in (torch.zeros_like(ids), torch.tensor([[0, 1, 1]]), torch.tensor([[1, 0, 1]])):
        with pytest.raises(ValueError):
            model(ids, mask)
    with pytest.raises(ValueError):
        model.embed(ids, ids, depth=2, truncate_dim=4)
    with pytest.raises(ValueError):
        model.embed(ids, ids, depth=3, truncate_dim=3)


def test_prefix_rounding_and_loss_grid():
    torch.manual_seed(3)
    raw = torch.randn(8, 4096)
    cached = prefix(raw, 384).half().float()
    torch.testing.assert_close(prefix(cached, 64), prefix(raw, 64), atol=0.0002, rtol=.001)
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/smoke.yaml')
    targets = torch.randn(4, 384)
    outputs = {d: targets.clone().requires_grad_() for d in cfg['model']['depths']}
    loss, logs = distillation(outputs, targets, [{'id':str(i)} for i in range(4)], cfg['model']['dims'], cfg)
    assert abs(float(loss.detach())) < 1e-6
    assert sum(k.endswith('/direct') for k in logs) == 25
    assert abs(sum(v for k,v in logs.items() if k.endswith('/weight')) - 1) < 1e-6
    loss.backward()
    assert all(z.grad is not None for z in outputs.values())


def test_group_split_transitive_before_dedup():
    rows = [dict(id=str(i), text_sha256=str(i), source_dataset='s', source_document_id=str(i)) for i in range(4)]
    rows[0]['parallel_group_id'] = rows[1]['parallel_group_id'] = 'p'
    rows[1]['source_document_id'] = rows[2]['source_document_id'] = 'doc'
    rows[3]['text_sha256'] = rows[2]['text_sha256']
    out = split_groups(rows, 42, .2, .2)
    assert len(out) == 3
    assert len({r['split_group'] for r in out}) == 1
    assert len({r['split'] for r in out}) == 1


def test_language_quotas_and_missing_coverage():
    sources = [{'language':'en','available_count':100000}, {'language':'es','available_count':100}]
    q, _ = quotas(sources, 100, 10, .3)
    assert sum(q.values()) == 100 and q['en'] <= 30 and q['es'] >= 10
    with pytest.raises(ValueError):
        quotas(sources, 10, 10, .3)


def test_atomic_cache_resume_checksums_and_ids(tmp_path):
    contract = {'corpus_id':'corpus'}
    rows = [{'id':'a','text_sha256':'hash-a'}, {'id':'b','text_sha256':'hash-b'}]
    m = open_cache(tmp_path, contract, False)
    values = np.random.default_rng(42).normal(size=(2, 384)).astype(np.float16)
    write_shard(tmp_path, m, 0, rows, {'targets':values})
    # An interrupted future shard has no completion marker and must be ignored.
    (tmp_path/'000000002.targets.npy.tmp').write_bytes(b'incomplete')
    m = open_cache(tmp_path, contract, True)
    assert len(m['shards']) == 1
    m['complete'] = True
    atomic_json(tmp_path/'manifest.json',m)
    cache = TargetCache(tmp_path, rows, 'corpus')
    np.testing.assert_array_equal(cache.batch(rows,'cpu').numpy(), values.astype(np.float32))
    with pytest.raises(ValueError):
        TargetCache(tmp_path,[{'id':'a','text_sha256':'wrong'}])
    with pytest.raises(ValueError):
        open_cache(tmp_path, {'corpus_id':'other'},True)
    (tmp_path/'000000000.targets.npy').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        open_cache(tmp_path,contract,True)


def test_sampler_resume_exact():
    rows = [{'id':str(i)} for i in range(11)]
    first = BatchStream(rows, 42)
    first.next(15)
    state = first.state()
    second = BatchStream(rows, 42)
    second.restore(state)
    assert first.next(25) == second.next(25)


def test_dynamic_onnx_loop_and_prefix_parity(tmp_path):
    pytest.importorskip('onnx')
    ort = pytest.importorskip('onnxruntime')
    from elastic_embed.deployment import ExportProfile, export_module
    model = small_model().eval()
    path = tmp_path/'profile.onnx'
    ids = torch.ones(1, 3, dtype=torch.long)
    export_module(ExportProfile(model,3),(ids,ids),path,['input_ids','attention_mask'])
    session = ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    for batch, length in ((1,1),(2,7),(1,65)):
        ids = torch.randint(0,32,(batch,length))
        mask = torch.ones_like(ids)
        if batch>1:
            mask[-1,3:]=0
        actual = torch.from_numpy(session.run(None,{'input_ids':ids.numpy(),'attention_mask':mask.numpy()})[0])
        with torch.no_grad():
            expected=model(ids,mask,(3,))[3]
        for dim in model.cfg['dims']:
            torch.testing.assert_close(prefix(actual,dim),prefix(expected,dim),atol=1e-4,rtol=1e-4)


def test_training_interruption_resume_matches_uninterrupted(tmp_path, monkeypatch):
    import elastic_embed.training as training
    cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/smoke.yaml')
    cfg['model'].update(backend='reference',vocab_size=32,input_rank=4,ffn_width=8)
    cfg['mohawk']['enabled'] = False
    cfg['training'].update(device='cpu',precision='float32',microbatch=1,effective_batch=2,
                           full_depth_steps=2,checkpoint_every=1,log_every=1,neighbor_fraction=0)
    def setup(path):
        c=copy.deepcopy(cfg); c['run_dir']=str(path); path.mkdir()
        rows=[dict(id=str(i),text_sha256=str(i),input_ids=[i+1,2],split='train',language='en') for i in range(4)]
        (path/'corpus.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        corpus={'id':'corpus','corpus_sha256':checksum(path/'corpus.jsonl'),'pad_id':0,
                'tokenizer_revision':c['teacher']['revision'],'preprocessing':'test'}
        atomic_json(path/'corpus_manifest.json',corpus)
        cachepath=path/'teacher_cache'
        m=open_cache(cachepath,cache_contract(c,corpus,'sentence'),False)
        values=np.random.default_rng(42).normal(size=(4,384)).astype(np.float16)
        write_shard(cachepath,m,0,rows,{'targets':values}); m['complete']=True
        atomic_json(cachepath/'manifest.json',m)
        seed_all(42); model=Student(c['model'])
        atomic_torch(path/'initialized.pt',{'model':model.state_dict(),'model_config':c['model'],'transferred':[]})
        return c
    baseline=setup(tmp_path/'baseline'); interrupted=setup(tmp_path/'interrupted')
    training.train(baseline,'full-depth')
    original=training.atomic_torch
    def stop_after_save(path,state):
        original(path,state)
        if state['step']==1:
            raise KeyboardInterrupt
    monkeypatch.setattr(training,'atomic_torch',stop_after_save)
    with pytest.raises(KeyboardInterrupt):
        training.train(interrupted,'full-depth')
    monkeypatch.setattr(training,'atomic_torch',original)
    training.train(interrupted,'full-depth',resume=True)
    a=torch.load(tmp_path/'baseline/full-depth.pt',weights_only=False,map_location='cpu')
    b=torch.load(tmp_path/'interrupted/full-depth.pt',weights_only=False,map_location='cpu')
    assert (a['step'],a['examples'],a['tokens'],a['sampler']) == (b['step'],b['examples'],b['tokens'],b['sampler'])
    for name in a['model']:
        torch.testing.assert_close(a['model'][name],b['model'][name],atol=0,rtol=0)
