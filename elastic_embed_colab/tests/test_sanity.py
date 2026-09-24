import pytest
import torch

from elastic_embed.sanity import fixture, score, evaluate_checkpoint
from elastic_embed.model import Student


def test_semantic_fixture_separates_correct_vectors_from_collapse():
    texts, cases, translations, candidates = fixture()
    vectors = torch.eye(len(texts))
    for row in [*cases, *translations]:
        vectors[row['query']] = vectors[row['positive']]
    good = score(vectors, cases, translations, candidates)
    assert good['easy_accuracy'] == good['hard_accuracy'] == 1
    assert good['retrieval']['hit@1'] == 1
    assert good['cross_language_retrieval']['hit@1'] == 1
    collapsed = score(torch.ones_like(vectors), cases, translations, candidates)
    assert collapsed['easy_accuracy'] == collapsed['hard_accuracy'] == 0
    assert collapsed['retrieval']['hit@1'] == collapsed['retrieval']['hit@5'] == 0
    assert any('collapse' in flag for flag in collapsed['flags'])
    vectors[0, 0] = float('nan')
    assert score(vectors, cases, translations, candidates)['status'] == 'nonfinite'


@pytest.mark.parametrize('stage', ['mohawk-block', 'full-depth', 'matryoshka'])
def test_explicit_checkpoint_encoding_without_teacher(tmp_path, monkeypatch, stage):
    import elastic_embed.sanity as sanity
    mc = dict(vocab_size=32, input_rank=8, width=16, layers=12, heads=4, head_dim=4,
              conv_size=4, ffn_width=24, backend='reference', depths=[1,3,6,9,12], dims=[2,4,8,12,16])
    cfg = dict(model=mc, teacher={'name':'test-tokenizer','revision':'test'}, training={'precision':'bfloat16'})
    path = tmp_path / f'{stage}.pt'
    model = Student(mc)
    torch.save({'model':model.state_dict(),'model_config':mc,'stage':stage,'step':2,'total_steps':2}, path)
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            ids = torch.tensor([[i%20+1,2,3] for i in range(len(texts))])
            return {'input_ids':ids,'attention_mask':torch.ones_like(ids)}
    monkeypatch.setattr(sanity,'tokenizer',lambda cfg: Tokenizer())
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        meta, vectors, dims = evaluate_checkpoint(cfg,path,['first','second','third'],'cpu','reference',2)
    finally:
        torch.set_num_threads(threads)
    assert meta['stage'] == stage and meta['checkpoint'] == str(path.resolve())
    assert meta['precision'] == 'float32' and meta['complete']
    assert len(meta['sha256']) == 64
    assert set(vectors) == set(mc['depths'])
    assert all(v.shape == (3,16) and torch.isfinite(v).all() for v in vectors.values())
