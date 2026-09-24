"""Cross-method invariants and provenance for the completed experiment."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / 'output/streaming_codecs'
data = json.loads((out / 'results.json').read_text())
rows = data['records']
assert len(rows) == 3960
assert len(data['runs']) == 216
groups = defaultdict(list)
phases = defaultdict(dict)
for r in rows:
    assert r['ids_preserved']
    assert r['posts'] == r['retained']
    assert r['packed_bytes'] == r['code_bytes'] + r['decoder_bytes'] + r['metadata_bytes']
    assert 0 <= r['local_proposal_recall10'] <= 1
    assert r['encode_seconds'] >= 0 and r['maintenance_seconds'] >= 0
    if r['method'] == 'Float32 reference':
        assert r['nrmse'] == r['membership_error'] == 0
        assert r['recall10'] == r['proposal_recall10'] == r['local_proposal_recall10'] == 1
    if r['phase'] == 'merged':
        assert r['leaves'] == 1 and r['membership_error'] == 0
    groups[r['case'], r['seed'], r['step'], r['phase']].append(r)
    phases[r['case'], r['seed'], r['step'], r['method']][r['phase']] = r
for rs in groups.values():
    assert len(rs) == 12
    assert len({(r['splits'], r['merges'], r['retained'], r['leaves'], r['tree_bytes']) for r in rs}) == 1
for key, pair in phases.items():
    if key[-1] != 'Sequential PPCA r4':
        for metric in ('nrmse', 'variance_ratio', 'recall10'):
            assert pair['grown'][metric] == pair['merged'][metric], (key, metric)
for run in data['runs']:
    assert run['expired_all_decoders']
    assert run['splits'] == run['merges']
    assert run['arrivals'] == (100000 if run['case'] == 'Bluesky 100,000' else 20480)
assert len(list(out.glob('*.bin'))) == 12
browser = json.loads((out / 'browser_validation.json').read_text())
assert not browser['errors'] and browser['desktop'] and browser['mobile']
tests = (out / 'tests.log').read_text()
assert 'Saved streaming codec payloads' in tests and 'ERROR' not in tests and 'Fail' not in tests
paths = ['src/StreamingCodecs.jl', 'examples/streaming_codec_experiment.jl',
         'examples/streaming_codec_viewer.html', 'scripts/streaming_codec_report.py',
         'test/streaming_codecs.jl', 'test/streaming_codec_outputs.jl', 'test/streaming_codec_viewer.cjs',
         'data/bluesky_100000/embeddings.f32', 'output/streaming_codecs/results.json',
         'output/streaming_codecs/demo.html']
hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
counts = [r for r in data['runs'] if r['method'] == 'Float32 reference']
validation = dict(records=len(rows), method_runs=len(data['runs']), tree_runs=len(counts),
                  splits=sum(r['splits'] for r in counts), merges=sum(r['merges'] for r in counts),
                  common_tree_invariants=True, immutable_codec_merge_invariants=True,
                  all_post_and_decoder_expiry=True, binary_payloads=12,
                  browser=browser, julia_tests=tests, sha256=hashes)
(out / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
print(json.dumps({k:v for k,v in validation.items() if k not in ('sha256','julia_tests')}, indent=2))
