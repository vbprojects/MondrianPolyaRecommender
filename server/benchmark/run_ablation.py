#!/usr/bin/env python3
"""Run engines sequentially, alternating order across seeds. Never replace a store."""
import json, os, pathlib, platform, subprocess, time, hashlib
root = pathlib.Path(__file__).resolve().parents[2]
out = root / 'output/server'
out.mkdir(parents=True, exist_ok=True)
config = {
    'seeds': [17, 29, 43], 'posts': 100_000, 'microbatch': 1024,
    'engine_order': [['sqlite', 'rocksdb'], ['rocksdb', 'sqlite'], ['sqlite', 'rocksdb']],
    'sqlite': {'journal': 'WAL', 'synchronous': 'FULL', 'cache_MiB': 32, 'mmap': False, 'schema': 'kv WITHOUT ROWID'},
    'rocksdb': {'sync': True, 'WAL': True, 'compression': 'none', 'block_cache_MiB': 16, 'write_buffer_MiB': 8, 'background_jobs': 2},
    'read_workload': '256 posting-slot lookups + 256 post-location lookups + first 25 term keys after every durable batch',
    'maintenance': '128-post split-like relocation, merge back, 1024 deletions; one atomic batch each',
    'cache_condition': 'warm OS cache, default engine caching; not a cold-device benchmark',
    'metadata_sources': ['data/keyword_experiment/yake/index.sqlite', 'data/disk_ablation/17-f16-append/index.sqlite'],
    'limits': ['metadata engine only, not alternate server backend', 'no vector/text IO or PPCA fitting', 'precomputed fixture/aggregate data in memory', 'cache knobs are not equal process memory caps', 'no weekly-scale extrapolation validation'],
    'platform': platform.platform(), 'cpu_count': os.cpu_count(),
    'cpu_model': next((line.split(':', 1)[1].strip() for line in pathlib.Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), ''),
    'julia': subprocess.check_output(['julia', '--version'], text=True).strip(),
    'manifest_sha256': hashlib.sha256((root/'server/benchmark/Manifest.toml').read_bytes()).hexdigest(),
    'script_sha256': hashlib.sha256((root/'server/benchmark/metadata_ablation.jl').read_bytes()).hexdigest(),
}
(out/'ablation-config.json').write_text(json.dumps(config, indent=2))
for seed, engines in zip(config['seeds'], config['engine_order']):
    for engine in engines:
        result = out/f'metadata-{engine}-{seed}.json'
        if result.exists():
            print(f'Already measured {engine} {seed}; leaving unchanged', flush=True)
            continue
        with (out/f'metadata-{engine}-{seed}.log').open('w') as log:
            start = time.monotonic()
            subprocess.run(['julia', '--threads=4,1', '--project=server/benchmark', 'server/benchmark/metadata_ablation.jl', engine, str(seed)], cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f'{engine} {seed}: {time.monotonic()-start:.1f}s process time', flush=True)
rows = [json.loads((out/f'metadata-{engine}-{seed}.json').read_text()) for seed in config['seeds'] for engine in ['sqlite', 'rocksdb']]
assert len({r['counts_digest'] for r in rows}) == 1
(out/'ablation-results.json').write_text(json.dumps(rows, indent=2))
