#!/usr/bin/env python3
"""Compare saved adapter baselines; never overwrite the original profile data."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/rocks_optimized'
OLD = ROOT / 'output/adapter_profile'


def read(path):
    return json.loads(path.read_text())


def summary(case):
    rows = [r['metrics'] for r in case['result']['results'] if r['kind'] == 'baseline']
    assert len(rows) == 3
    assert statistics.median(r['compile_seconds'] for r in rows) == 0
    assert all(r['recompile_seconds'] == 0 for r in rows)
    return {k: statistics.median(r[k] for r in rows) for k in rows[0]}


cases = {}
compilation = []
for workload in ('ingestion', 'terms'):
    for backend in ('rocksdb', 'sqlite'):
        name = f'{backend}-{workload}'
        before, after = read(OLD / f'{name}.json'), read(OUT / f'{name}.json')
        for i, row in enumerate(after['result']['results'], 1):
            if row['kind'] == 'baseline' and row['metrics']['compile_seconds']:
                compilation.append(f"{name} run {i}: {1000*row['metrics']['compile_seconds']:.1f} ms")
        digest = 'topology_digest' if workload == 'ingestion' else 'term_digest'
        assert before['result'][digest] == after['result'][digest], name
        cases[name] = {'before': summary(before), 'after': summary(after)}
        cases[name]['speedup'] = cases[name]['before']['seconds'] / cases[name]['after']['seconds']
        cases[name]['allocation_reduction'] = 1 - cases[name]['after']['allocated_bytes'] / cases[name]['before']['allocated_bytes']
for workload in ('ingestion', 'terms'):
    digest = 'topology_digest' if workload == 'ingestion' else 'term_digest'
    assert read(OUT / f'rocksdb-{workload}.json')['result'][digest] == read(OUT / f'sqlite-{workload}.json')['result'][digest]
sources = [read(OUT / f'{name}.json')['sources'] for name in cases]
assert all(s == sources[0] for s in sources)
stats = read(OUT / 'rocksdb-terms.json')['result']['storage']
validation = read(OUT / 'validation.json')
legacy = read(OUT / 'legacy_rows/rocksdb-terms.json')
assert legacy['result']['term_digest'] == read(OUT / 'rocksdb-terms.json')['result']['term_digest']
assert legacy['result']['storage']['rocksdb_actual']['format_version'].endswith('-v1')
data = {'cases': cases, 'compact_storage': stats, 'legacy': summary(legacy), 'validation': validation}
(OUT / 'comparison.json').write_text(json.dumps(data, indent=2) + '\n')
lines = ['# RocksDB adapter optimizations', '',
    'Implemented compact metadata rows, specialized hot operations, direct key construction, unchanged-index elimination, and a common-keyword query path that skips global-count reads and log-lift. The undo journal, durable commits, Float16 payloads and public API remain unchanged.', '',
    '## Measured before and after', '',
    'Median of three unprofiled sequential runs after workload warmup and GC, in separate processes. Median reported compilation time is zero. Nonzero compilation samples, retained in timing: ' + ('; '.join(compilation) or 'none') + '. The before values are the saved adapter profile baselines; the after values are a fresh run with the same fixtures. This is a combined optimization result, not a causal ablation of each change.', '',
    '| Workload | Before | After | Speedup | Allocated MiB before → after |',
    '| --- | ---: | ---: | ---: | ---: |']
for name, c in cases.items():
    b, a = c['before'], c['after']
    lines.append(f"| {name} | {b['seconds']:.3f} s | {a['seconds']:.3f} s | {c['speedup']:.2f}× | {b['allocated_bytes']/2**20:,.1f} → {a['allocated_bytes']/2**20:,.1f} |")
r = cases['rocksdb-ingestion']; q = cases['rocksdb-terms']; s = cases['sqlite-ingestion']
lines += ['',
    f"RocksDB ingestion improves from {2048/r['before']['seconds']:,.0f} to {2048/r['after']['seconds']:,.0f} posts/s. SQLite now measures {2048/s['after']['seconds']:,.0f} posts/s. Common-term requests average {q['after']['seconds']*1000/6:.0f} ms within the median six-request run, down from {q['before']['seconds']*1000/6:.0f} ms. These are not latency percentiles.", '',
    '**Allocation volume is cumulative Julia allocation, not peak RAM.** Native memory allocation and physical SSD IO are not measured by this harness.', '',
    '- Ingestion: 2,048 normalized synthetic 384D vectors, seed 82, eight durable 256-post batches, Float16, rank-4 PPCA, splits, prepared hashtags and committed source cursors. Embedding inference and keyword extraction are excluded.',
    '- Terms: six direct server common-term calls against the same 100,000-post GTE-tiny fixture, five leaves, generation 102, fixed historical clock. The optimized RocksDB fixture is a new compact-format copy. These calls scan the full leaf vocabulary and bypass HTTP, owner queue and networking.',
    '- Same 32 MiB managed budget with 4 MiB allocated to metadata; warm OS caches, no cache dropping. Four Julia default threads, one BLAS thread. Workstation: AMD Ryzen 7 7800X3D, 16 logical CPUs exposed by the host.',
    '- SQLite also benefits from skipping the unnecessary common-term global join. Distinctive ranking still reads global counts and is correctness-tested, but has no speed claim from these timings.', '',
    '## Existing records and storage', '',
    f"The new query path on the existing v1 RocksDB fixture takes {data['legacy']['seconds']:.3f} s for six requests, without rewriting records or upgrading its format marker. Compact rows take {q['after']['seconds']:.3f} s. This separates the practical read-only deployment case from a fresh compact store, although the two stores have different SST histories.", '',
    f"The new 100k copy occupies {stats['metadata_bytes']/2**20:.1f} MiB of metadata and {stats['total_bytes']/2**20:.1f} MiB including payloads at measurement time. Vector and text files total {(stats['vector_files']+stats['text_files'])/2**20:.1f} MiB and were copied unchanged. Metadata sizes include engine files and depend on compaction history; this is not a normalized write-amplification or compression-ratio study.", '',
    'The reader supports old serialized and new fixed-schema rows in the same store. The first write transaction upgrades the store format marker atomically; unchanged old rows are not automatically rewritten. Older binaries reject upgraded stores. Keep an offline pre-upgrade backup when rollback to an older binary is required. No live store was migrated or switched.', '',
    '## Correctness', '',
    'All before/after and SQLite/RocksDB topology and returned-term digests match. Every synthetic ingestion run passes the storage audit; every terms run preserves its committed generation. The import verifies identities, payload hashes, topology and posting integrity.', '',
    f"Recorded validation: {validation['assertions_passed']:,} passing assertions across {len(validation['suites'])} suite runs. Coverage includes wire-format round trips, ordered key compatibility, mixed old/new records, rollback of format upgrades, forced eviction, structural changes, keywords, live ingestion, HTTP, and process-crash recovery. See [validation.json](../../output/rocks_optimized/validation.json) for individual suites.", '',
    '## Recommendation', '',
    'Use the optimized adapter for further RocksDB experiments, while retaining SQLite as the default until sustained ingestion, compaction and concurrent server workloads establish the tradeoff. Keyword queries still scan a leaf vocabulary: a bounded summary cache or maintained top-term summaries are separate future work. These measurements do not establish an inherent LSM advantage or a production capacity guarantee.', '',
    '## Reproduce', '',
    'Run from the repository root with the private 100k fixture available; use a new destination for the import:', '',
    '```sh',
    'julia --threads=4,1 --project=server server/bin/import_sqlite.jl --source data/server_http_pilot --destination data/rocks_optimized_100k --cache-mib 32',
    'export MONDRIAN_PROFILE_OUTPUT=output/rocks_optimized',
    'export MONDRIAN_PROFILE_ROCKS_STORE=data/rocks_optimized_100k',
    'export MONDRIAN_PROFILE_BASELINE_ONLY=1',
    'for workload in ingestion terms; do',
    '  for backend in rocksdb sqlite; do',
    '    julia --threads=4,1 --project=server server/benchmark/profile_adapter.jl "$backend" "$workload"',
    '  done',
    'done',
    'MONDRIAN_PROFILE_OUTPUT=output/rocks_optimized/legacy_rows MONDRIAN_PROFILE_ROCKS_STORE=data/rocks_backend_100k julia --threads=4,1 --project=server server/benchmark/profile_adapter.jl rocksdb terms',
    'julia --threads=4,1 --project=server server/test/rocks_optimized.jl',
    'python server/benchmark/optimized_report.py',
    '```', '',
    '[Interactive comparison](../../output/rocks_optimized/demo.html) · [Original profiles](Adapter_Profile.md) · [Backend and compatibility details](../RocksDB.md)', '']
(ROOT / 'server/benchmark/RocksDB_Optimizations.md').write_text('\n'.join(lines))
html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RocksDB adapter optimizations</title>
<style>body{font:16px system-ui;max-width:1050px;margin:32px auto;padding:0 20px;background:#101923;color:#e4ebf2}section{overflow-x:auto;background:#192737;padding:20px;border-radius:12px;margin:20px 0}a{color:#93caff}table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #435367;padding:12px}select{font:inherit;padding:8px;margin:6px;background:#26384c;color:inherit}.bar{height:22px;border-radius:4px;background:#70cbb8}.note{color:#b4c4d4;line-height:1.6}</style>
<h1>RocksDB adapter optimizations</h1><p>Same tree, payloads and durability. Compact metadata, fewer index writes, and fewer keyword lookups.</p>
<section><h2>Before and after</h2><table id="results"></table><p class="note">Median of three warmed runs. Ingestion covers 2,048 posts; terms covers six direct server queries against 100,000 posts. Allocations are cumulative, not resident memory.</p></section>
<section><h2>Compare a workload</h2><label>Workload <select id="workload"><option value="ingestion">Durable ingestion</option><option value="terms">Common keyword queries</option></select></label><label>Metric <select id="metric"><option value="seconds">Elapsed seconds</option><option value="allocated_bytes">Allocated MiB</option></select></label><div id="bars"></div><p class="note">Lower is better. Before measurements are archived profiles; after measurements use the optimized code. Warm OS caches. No embedding inference or keyword extraction.</p></section>
<section><h2>Compatibility and limits</h2><p id="legacy"></p><p>Existing records remain readable. The first write upgrades the format marker; older binaries then require a pre-upgrade backup. New compact records are written incrementally. No live store was switched.</p><p id="validation"></p><p>Retain SQLite as the default pending sustained ingestion and concurrent server tests. This is not a general B-tree versus LSM verdict.</p></section>
<p><a href="../../server/benchmark/RocksDB_Optimizations.md">Report and reproduction</a> · <a href="comparison.json">Comparison data</a> · <a href="validation.json">Validation</a> · <a href="config.json">Configuration</a> · <a href="../adapter_profile/demo.html">Original profiles</a></p>
<script>const data=__DATA__;const names=Object.keys(data.cases);document.getElementById('results').innerHTML='<tr><th>Case</th><th>Before</th><th>After</th><th>Speedup</th></tr>'+names.map(n=>{const c=data.cases[n];return '<tr><td><a href="'+n+'.json">'+n+'</a></td><td>'+c.before.seconds.toFixed(3)+' s</td><td>'+c.after.seconds.toFixed(3)+' s</td><td>'+c.speedup.toFixed(2)+'×</td></tr>'}).join('');function render(){const w=document.getElementById('workload').value,m=document.getElementById('metric').value;const rows=['rocksdb','sqlite'].flatMap(b=>['before','after'].map(v=>({label:b+' '+v,value:data.cases[b+'-'+w][v][m]/(m==='allocated_bytes'?1048576:1)})));const max=Math.max(...rows.map(r=>r.value));document.getElementById('bars').innerHTML=rows.map(r=>'<p>'+r.label+': '+r.value.toFixed(3)+'</p><div class="bar" style="width:'+100*r.value/max+'%"></div>').join('')}document.getElementById('workload').onchange=render;document.getElementById('metric').onchange=render;render();document.getElementById('legacy').textContent='Six requests with old rows: '+data.legacy.seconds.toFixed(3)+' s. With compact rows: '+data.cases['rocksdb-terms'].after.seconds.toFixed(3)+' s. Payloads are unchanged.';document.getElementById('validation').textContent=data.validation.assertions_passed.toLocaleString()+' assertions passed; topology and term digests match before/after and across backends.';</script></html>'''
(OUT / 'demo.html').write_text(html.replace('__DATA__', json.dumps(data).replace('<', '\\u003c')))
print('Wrote RocksDB_Optimizations.md and output/rocks_optimized/demo.html')
