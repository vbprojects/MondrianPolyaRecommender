#!/usr/bin/env python3
"""Render adapter profiles. Timing baselines and sampled profiles stay separate."""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/adapter_profile'
cases = {f'{b}-{w}': json.loads((OUT / f'{b}-{w}.json').read_text())
         for w in ('ingestion', 'terms') for b in ('sqlite', 'rocksdb')}

def profile(case, kind):
    return next(r['metrics']['profile'] for r in case['result']['results'] if r['kind'] == kind)

def baseline(case):
    rows = [r['metrics'] for r in case['result']['results'] if r['kind'] == 'baseline']
    assert len(rows) == 3
    assert all(r['compile_seconds'] == 0 and r['recompile_seconds'] == 0 for r in rows)
    return {key: st.median(r[key] for r in rows) for key in rows[0]}

assert cases['sqlite-ingestion']['result']['topology_digest'] == cases['rocksdb-ingestion']['result']['topology_digest']
assert cases['sqlite-terms']['result']['term_digest'] == cases['rocksdb-terms']['result']['term_digest']
for path in ('src/RocksMetadata.jl', 'src/DiskTreeStore.jl', 'src/DiskKeywords.jl', 'server/src/Terms.jl', 'server/Manifest.toml'):
    assert len({case['sources'][path] for case in cases.values()}) == 1, path
for case in cases.values():
    p = profile(case, 'cpu')
    assert sum(r['weight'] for r in p['exclusive']) == p['total_weight'] > 0
    assert abs(sum(r['percent'] for r in p['exclusive']) - 100) < 1e-8

def share(case, category, kind='exclusive'):
    return next((r['percent'] for r in profile(case, 'cpu')[kind] if r['category'] == category), 0.)

labels = {
    'other_adapter': 'Row/index handling and adapter iteration',
    'serialization': 'Serialization/deserialization',
    'statement_adapter': 'Statement dispatch, parsing and projection',
    'key_encoding': 'Key encoding',
    'rocksdb_binding_native': 'RocksDB binding/native path',
    'sqlite_binding_native': 'SQLite binding/native path',
    'journal_orchestration': 'Journal orchestration (exclusive)',
    'keywords_and_ranking': 'Keyword aggregation and ranking',
    'density_and_linear_algebra': 'Density / linear algebra',
    'other_tree_payload': 'Other tree / payload work',
    'unattributed': 'Unattributed',
}
def native_paths(name):
    d = json.loads((OUT / f'{name}-cpu.speedscope.json').read_text())
    frames = d['shared']['frames']; p = d['profiles'][0]; total = sum(p['weights'])
    predicates = {
        'Native RocksDB library frame': lambda f: 'librocksdb' in f['file'],
        'rocksdb_get call path': lambda f: f['name'] == 'rocksdb_get',
        'rocksdb_write call path': lambda f: f['name'] == 'rocksdb_write',
        'rocksdb_iter_* call path': lambda f: f['name'].startswith('rocksdb_iter_'),
    }
    return {key: 100 * sum(w for s, w in zip(p['samples'], p['weights'])
                          if any(pred(frames[i]) for i in s)) / total
            for key, pred in predicates.items()}
native = {w: native_paths('rocksdb-'+w) for w in ('ingestion', 'terms')}
lines = ['# Adapter profile', '',
    'The main ingestion cost is in adapter record/index handling, serialization and statement dispatch. Keyword reads split much of their sampled execution between deserialization and native point reads. The foreground journal is a smaller contributor in this fixture than initially suspected. The storage implementation was left unchanged; raw stack profiles and allocation samples are included.', '',
    '## Unprofiled baselines', '',
    'Median of three sequential runs per case, each preceded by a full workload warmup and garbage collection. These timed runs have no sampling profiler enabled. All reported compilation/recompilation times were zero. Embedding inference and keyword extraction are excluded.', '',
    '| Case | Time per workload | Julia allocation volume | GC time |', '| --- | ---: | ---: | ---: |']
for name, case in cases.items():
    b = baseline(case)
    lines.append(f"| {name} | {b['seconds']:.3f} s | {b['allocated_bytes']/1024**2:,.1f} MiB | {b['gc_seconds']:.3f} s |")
ri, si = (baseline(cases[b+'-ingestion']) for b in ('rocksdb', 'sqlite'))
rt, st_ = (baseline(cases[b+'-terms']) for b in ('rocksdb', 'sqlite'))
lines += ['',
    f'Ingestion covers 2,048 synthetic normalized 384D posts in eight durable batches of 256, with Float16 storage, rank-4 PPCA, splits, prepared hashtag contributions and cursor commits. Throughput: SQLite {2048/si["seconds"]:,.0f} posts/s; RocksDB {2048/ri["seconds"]:,.0f} posts/s. Every run passed the storage audit and produced the same topology, including across engines.', '',
    'The terms workload makes six direct calls to the actual server term-ranking function, using vectors from the first six active IDs in the same 100,000-post historical GTE-tiny fixture. It bypasses HTTP, networking and the owner queue. Each request scans the full vocabulary of its routed leaf; this is not a cached label lookup. Fixed historical clock; no ingestion or inference runs concurrently. Both engines returned identical leaf labels/counts/scores and preserved the committed generation.', '',
    f'Mean time per request within the median six-query run: SQLite {st_["seconds"]/6*1000:.0f} ms; RocksDB {rt["seconds"]/6*1000:.0f} ms. These are not request-latency percentiles.', '',
    '**Allocation volume is cumulative Julia-managed allocation, not resident RAM or native RocksDB memory.** Native malloc activity is not captured by Julia’s allocation profiler. Most allocated objects can be reclaimed during the workload.', '',
    '## Where sampled execution landed', '',
    'The following shares assign each captured workload stack to its nearest recognized subsystem, starting at the executing end of the stack. Base/runtime helpers are charged to their recognized caller. Columns sum to 100% (rounding aside). They are sampled stack shares, not a precise wall-time or CPU-cycle accounting. Raw C and Julia frames are retained for inspection.', '',
    '| Exclusive attribution | RocksDB ingestion | RocksDB terms |', '| --- | ---: | ---: |']
for cat, label in labels.items():
    vals = [share(cases['rocksdb-'+w], cat) for w in ('ingestion', 'terms')]
    if any(vals): lines.append(f'| {label} | {vals[0]:.1f}% | {vals[1]:.1f}% |')
lines += ['',
    f'The entire journal call path appears in {share(cases["rocksdb-ingestion"],"journal_orchestration","inclusive"):.1f}% of ingestion samples. This **inclusive** figure includes nested native calls and serialization, and must not be added to the table above. The journal can also create later background compaction work that this number does not account for.', '',
    'Within the native paths, the recorded symbols show:', '',
    '| Inclusive call path | Ingestion samples | Terms samples |', '| --- | ---: | ---: |']
for label in native['ingestion']:
    lines.append(f'| {label} | {native["ingestion"][label]:.1f}% | {native["terms"][label]:.1f}% |')
lines += ['', 'These paths overlap each other and the preceding categories. They count stacks containing the named library/API frame, not calls or exact native elapsed time. Native query cost is mainly on the point-read path in this capture.', '',
    '## Allocation sites', '',
    'Allocation profiling ran separately, sampling allocation events with probability 0.0001. The rows below show the largest shares of sampled bytes; rare large allocations can make these estimates noisy. Use the exact baseline totals above for allocation volume.', '']
for workload in ('ingestion', 'terms'):
    p = profile(cases['rocksdb-'+workload], 'alloc')
    lines += [f'### RocksDB {workload}', '', f'{p["sampled_allocations"]:,} sampled allocation events.', '', '| Site | Share of sampled bytes |', '| --- | ---: |']
    for row in p['bytes']['top_sites'][:8]:
        lines.append(f'| `{row["category"].replace(str(ROOT)+"/", "")}` | {row["percent"]:.1f}% |')
    lines.append('')
lines += [
    '## Interpretation and next step', '',
    '1. **Benchmark compact, fixed-schema metadata encoding first.** Deserializing NamedTuple type information and symbols is prominent in both profiles. This concerns metadata rows, not Float16 vector compression. Use a versioned experimental store so existing records remain readable.',
    '2. **Reduce repeated row/index construction and string-statement work on ingestion.** `indexkeys`, `normalize_row`, key construction and statement dispatch are visible hotspots. Typed operations and updating only affected indexes should be tested independently.',
    '3. **Remove unnecessary point reads from common-term queries.** Code inspection shows the common ranking still fetches each term’s global count and computes log-lift, although its score uses only leaf support. The profile confirms point-read/deserialization cost, but does not quantify the speedup from removing this work. Distinctive ranking still needs the global data.',
    '4. **Defer transaction-protocol redesign until those costs are measured again.** The journal’s inclusive share limits how much eliminating its foreground work alone could help this fixture. Its indirect background work and larger structural migrations still need separate measurements.', '',
    'GC pause time is a small part of the baseline duration. Allocation churn matters here through object construction, type reconstruction and copying as well as collection; simply changing GC settings would not address most of that work.', '',
    'The profiles do not establish a general B-tree/LSM winner. RocksDB compaction/flush threads are native background threads and are not fully captured by Julia’s profiler; OS-cache reads and waits can also appear under native calls. A native-call sample is not proof of physical SSD IO. We did not count fsync calls or attribute SSD write amplification in this experiment.', '',
    '## Reproduce and inspect', '',
    'Run from the repository root, sequentially (no competing benchmark processes):', '', '```sh',
    'julia --threads=4,1 --project=server server/benchmark/profile_adapter.jl rocksdb ingestion',
    'julia --threads=4,1 --project=server server/benchmark/profile_adapter.jl sqlite ingestion',
    'julia --threads=4,1 --project=server server/benchmark/profile_adapter.jl rocksdb terms',
    'julia --threads=4,1 --project=server server/benchmark/profile_adapter.jl sqlite terms',
    'python server/benchmark/adapter_profile_report.py', '```', '',
    '- Existing private fixtures: `data/rocks_backend_100k` and `data/server_http_pilot`. The script requires an existing committed store for terms. Synthetic ingestion uses temporary stores.',
    '- Same 32 MiB managed store budget, 4 MiB metadata allowance, and warm OS caches as the earlier comparison. Four Julia default threads; BLAS one thread. Each case runs in a fresh process.',
    '- Time profiling: nominal 1 ms sampling, workload-stack filter excludes unrelated/idle Julia threads. Instrumented durations are stored separately and are not benchmark throughput. Sampling cadence and GC/safepoints can bias shares.',
    '- The initial ingestion captures were reclassified from their saved raw stacks to avoid attributing generic Julia serialization methods to the SQLite precompile image. The correction changes analysis only. Analysis and measured-source hashes are recorded.',
    '- `*-cpu.speedscope.json`: time-sample call stacks, weighted by sample count. `*-alloc.speedscope.json`: sampled allocation stacks weighted by observed bytes. Load these files into a Speedscope-compatible viewer; they contain no post text or vectors.',
    '- `*-cpu-flat.txt`: standard Julia profiler output, including C frames and other threads. `*.json`: baseline runs, category summaries, source hashes and correctness checks.', '',
    '[Interactive summary](../../output/adapter_profile/demo.html). No storage format, transaction protocol or live configuration was changed by this experiment.', '']
(ROOT/'server/benchmark/Adapter_Profile.md').write_text('\n'.join(lines))

html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Storage adapter profile</title>
<style>body{font:16px system-ui;background:#101824;color:#e4edf5;max-width:1100px;margin:32px auto;padding:0 24px}p{line-height:1.6}section{background:#1b2a3b;padding:20px;border-radius:10px;margin:20px 0}a{color:#9acfff}select{padding:8px;font:inherit}table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #455264}.bar{height:18px;background:#67c7b3;margin:5px 0 12px}pre{white-space:pre-wrap}.note{color:#ffd196}</style>
<h1>Storage adapter profile</h1><p><a href="../rocks_optimized/demo.html">Follow-up: implemented optimizations and measurements</a></p><p>Unchanged implementation. Warmed, sequential baselines and separate time/allocation profiles. Synthetic 2,048-post ingestion; six real-fixture keyword queries. No inference or extraction in the measured workloads.</p>
<section><h2>Unprofiled baselines</h2><table id="baseline"></table><p class="note">Allocation is cumulative Julia allocation volume, not peak RAM. Terms timing covers six queries. Three runs per case; median shown.</p></section>
<section><h2>Inspect a profile</h2><select id="case"></select> <select id="kind"><option value="cpu">Execution sample shares</option><option value="alloc">Sampled allocation bytes</option></select><div id="bars"></div><p class="note">Exclusive attribution to the nearest recognized subsystem. Native background compaction threads are not fully captured. Allocation samples are sparse; small differences are not meaningful.</p><div id="links"></div></section>
<p><a href="../../server/benchmark/Adapter_Profile.md">Detailed report, limitations and reproduction</a> · <a href="../rocks_backend/demo.html">Full-backend comparison</a></p>
<script>const data=__DATA__,labels=__LABELS__;const median=a=>a.sort((x,y)=>x-y)[Math.floor(a.length/2)];const entries=Object.entries(data);const c=document.getElementById('case');c.innerHTML=entries.map(([k])=>'<option>'+k+'</option>').join('');c.value='rocksdb-ingestion';document.getElementById('baseline').innerHTML='<tr><th>Case</th><th>Seconds</th><th>Allocated MiB</th><th>GC seconds</th></tr>'+entries.map(([k,v])=>{const a=v.result.results.filter(r=>r.kind==='baseline').map(r=>r.metrics);return '<tr><td>'+k+'</td>'+['seconds','allocated_bytes','gc_seconds'].map(n=>'<td>'+(median(a.map(r=>r[n]))/(n==='allocated_bytes'?1048576:1)).toFixed(3)+'</td>').join('')+'</tr>'}).join('');function render(){const name=c.value,kind=document.getElementById('kind').value;const p=data[name].result.results.find(r=>r.kind===kind).metrics.profile;const rows=kind==='cpu'?p.exclusive:p.bytes.exclusive;document.getElementById('bars').innerHTML=rows.map(r=>'<p>'+(labels[r.category]||r.category)+': '+r.percent.toFixed(1)+'%</p><div class="bar" style="width:'+r.percent+'%"></div>').join('');document.getElementById('links').innerHTML='<a href="'+name+'.json">Raw measurements</a> · <a href="'+name+'-'+kind+'.speedscope.json">Download stack profile</a>'}c.onchange=render;document.getElementById('kind').onchange=render;render();</script></html>'''
(OUT/'demo.html').write_text(html.replace('__DATA__', json.dumps(cases)).replace('__LABELS__', json.dumps(labels)))
print('Wrote Adapter_Profile.md and output/adapter_profile/demo.html')
