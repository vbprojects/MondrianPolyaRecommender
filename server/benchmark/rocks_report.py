#!/usr/bin/env python3
"""Render the integrated backend measurements without rerunning the workload."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/rocks_backend'
read = lambda name: json.loads((OUT / name).read_text())
pilots = {b: read(f'http-{b}.json') for b in ('sqlite', 'rocksdb')}
ingestion = {r['backend']: r for r in read('ingestion.json')['results']}
validation = read('validation.json')
live = read('live-restart.json')
mib = lambda x: x / 1024**2

lines = [
    '# Integrated RocksDB backend results', '',
    'RocksDB is now selectable for the complete disk tree and HTTP server. The existing live store was not switched. This prototype passes lifecycle, recovery and cross-backend parity checks, but these full-backend measurements favor SQLite for ingestion and metadata size. The earlier metadata-only 6× result does not carry over to this implementation.', '',
    '## 100,000 real posts', '',
    'An explicit offline import copied the existing historical GTE-tiny fixture: 100,000 Float16 384D vectors, five leaves, 346,145 dictionary terms and 732,333 leaf/global term-count records. Payload hashes, topology and generation were preserved; no vectors were re-encoded. The destination has no SQLite metadata database.', '',
    'Each engine served 12 requests per endpoint after warmup, sequentially by engine, with the same source vectors and historical clock. Connections are fresh for each request to avoid idle pool expiry across compilation warmup. Regional candidate sampling remains randomized. These are small, single-client, warm-OS-cache measurements, not cold-device results, a multi-seed performance ablation, or a latency SLO. The pilots did not run ingestion or inference, but background correctness validation was active during part of the workstation runs; timings are indicative rather than isolated performance estimates.', '',
    '| Endpoint | SQLite p50 / p95 | RocksDB p50 / p95 |',
    '| --- | ---: | ---: |',
]
for name in ('tree', 'nearby', 'terms'):
    cells = [f"{pilots[b]['timings'][name]['p50_ms']:.1f} / {pilots[b]['timings'][name]['p95_ms']:.1f} ms" for b in pilots]
    lines.append(f'| /v1/{name} | ' + ' | '.join(cells) + ' |')
lines += ['', '| Footprint at measurement | SQLite | RocksDB |', '| --- | ---: | ---: |']
for label, values in (
    ('Metadata files, including engine logs', [p['storage_stats']['metadata_bytes'] for p in pilots.values()]),
    ('All store files', [p['storage_stats']['total_bytes'] for p in pilots.values()]),
    ('Process peak RSS, includes Julia/runtime and HTTP client', [p['process_memory']['VmHWM'] for p in pilots.values()]),
):
    lines.append(f'| {label} | ' + ' | '.join(f'{mib(v):.1f} MiB' for v in values) + ' |')
lines += [
    '', 'All nearby responses returned 25 IDs and reads left the generation unchanged. Both engines share the same scattered vector blocks and full leaf-vocabulary ranking. Those costs remain regardless of metadata engine. Reported vector file reads may be served by the OS cache; they are not SSD device traffic.', '',
    'The managed store budget was 32 MiB, including a 4 MiB database allowance, plus a separate 32 MiB server-buffer allowance. RocksDB uses a 2 MiB block cache (also for partitioned SST index/filter blocks), up to two 1 MiB memtables and a 512 KiB application write-buffer target. These allowances are not process RSS limits. Native usage is in the raw results.', '',
    'RocksDB general compression is disabled. Verbose serialized records, ordered secondary indexes, WAL/compaction state and the undo journal affect its footprint. The RocksDB pilot received an explicit full SST rewrite before measurement to apply partitioned indexes/filters to the initially imported files; SQLite did not receive VACUUM. File sizes describe these actual stores, not an equal-reclamation or inherent B-tree versus LSM ratio.', '',
    '## Integrated ingestion smoke measurement', '',
    '2,048 seeded synthetic normalized 384D vectors, prepared hashtag contributions, eight durable batches of 256, Float16 storage and the common rank-4 PPCA tree. A 256-post warmup preceded each engine. Embedding inference and keyword preparation are excluded. Both produced the same 25-leaf tree.', '',
    '| Engine | Durable storage/tree ingestion | Total measured time |', '| --- | ---: | ---: |',
]
for b, r in ingestion.items():
    lines.append(f"| {b} | {r['posts_per_second']:.0f} posts/s | {r['seconds']:.3f} s |")
lines += [
    '', 'Both clear 100 posts/s in this small fixture. This is one smoke run on a Ryzen 7 7800X3D host, with four Julia worker threads and BLAS limited to one thread; it does not establish sustained live capacity. Recovery tests ran separately as background validation on this workstation during the measurement period, so timing differences should be treated as indicative. The RocksDB path includes application statement dispatch, secondary-index maintenance and an undo journal absent from the earlier microbenchmark.', '',
    '## Correctness', '',
]
for suite in validation['suites']:
    lines.append(f"- {suite['name']}: {suite['passed']} assertions passed.")
lines += [
    '', 'Checks cover source cursor publication, keyword counts, splits/merges, edits/deletion/expiry, compaction, restart/eviction, API semantics, exact decoded-vector parity and explicit import. Subprocess crashes cover open/committed transactions and interruption during recovery itself. These are process-kill tests, not physical power-loss or filesystem-fault certification.', '',
    f'A separate bounded combined-mode run collected two 64-event live Bluesky batches using the existing 256D encoder and Julia YAKE, answered HTTP health, and shut down cleanly. Reopening audited {live["restart_audit_posts"]} active posts with {live["keywords"]["coverage"]:.0%} keyword coverage and {live["cursor_count"]} persisted cursor records. This is connectivity/restart validation, not a sustained ingestion benchmark. Raw records are in `live-smoke.json` and `live-restart.json`.', '',
    '## Recommendation', '',
    'Keep SQLite as the default while testing RocksDB explicitly. The native backend is functional; it is not currently an efficiency upgrade for this full workload. An initial unpartitioned SST configuration thrashed its tiny index/filter cache (terabytes of OS-cache reads); that run was stopped. Partitioned indexes and filters are now the default, keeping metadata fetches small. Next optimization candidates are typed operations in the hot path, compact record encoding, metadata compression and reducing journal/secondary-index writes. Vector locality and cached or incrementally maintained term rankings are separate shared query improvements.', '',
    'RocksDB still writes a WAL, SST files, compactions and the application undo journal. Leaf locality does not eliminate write amplification. The journal is the current tradeoff for bounded-memory, crash-safe atomic structural updates under one store owner.', '',
    'See [../RocksDB.md](../RocksDB.md) for run/import commands and the single-owner visibility contract. [Interactive measurements](../../output/rocks_backend/demo.html) and raw JSON are in `output/rocks_backend/`; configuration and both environments are pinned. Million-post, seven-day, sustained mixed-query/ingestion capacity and cold-device performance remain unmeasured.', '',
]
(ROOT / 'server/benchmark/RocksDB_Results.md').write_text('\n'.join(lines))

data = json.dumps({'pilots': pilots, 'ingestion': ingestion})
html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Integrated RocksDB backend</title><style>
body{font:16px system-ui;background:#101824;color:#e4edf5;max-width:1040px;margin:32px auto;padding:0 24px}p{line-height:1.6}section{background:#1b2a3b;padding:20px;border-radius:12px;margin:20px 0}a{color:#9acfff}select{padding:8px;font:inherit}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #455264;padding:12px}.bar{height:22px;border-radius:4px;margin:10px 0 20px}.note{color:#ffd196}
</style><h1>RocksDB: the complete backend</h1><p><a href="../adapter_profile/demo.html">Adapter execution and allocation profiles</a></p>
<p>Selectable metadata engine, shared Float16 payloads, PPCA tree, keywords and HTTP API. SQLite remains the default. The existing live index was not switched.</p>
<section><h2>100,000 real posts · HTTP</h2><p>Five leaves, 384D GTE-tiny. Twelve requests per endpoint after warmup; single client and warm OS cache. Both engines returned 25 nearby IDs and preserved the generation.</p><table id="http"></table></section>
<section><h2>Storage and ingestion</h2><label>Measure <select id="metric"><option value="metadata">100k metadata MiB — lower is better</option><option value="total">100k all store files MiB — lower is better</option><option value="ingest">2,048-post synthetic ingestion posts/s — higher is better</option><option value="rss">100k pilot process peak RSS MiB — lower is better</option></select></label><div id="bars"></div><p class="note">Ingestion includes durable commits and tree/keyword updates; inference and keyword preparation are excluded. One small smoke run, not sustained capacity. File sizes include engine logs and differ in compaction state. Managed cache budgets are not RSS limits.</p></section>
<section><h2>What changed</h2><p>Native ordered indexes, bounded caches, disk-backed undo journal, interrupted-recovery tests and explicit SQLite import. The full implementation incurs costs that were absent from the earlier metadata-only benchmark. Current measurements favor keeping SQLite as the default while optimizing the RocksDB path.</p><p><a href="../../server/benchmark/RocksDB_Results.md">Full results and limitations</a> · <a href="../../server/RocksDB.md">Run and import commands</a> · <a href="config.json">Configuration</a> · <a href="validation.json">Validation</a> · <a href="http-rocksdb.json">RocksDB HTTP</a> · <a href="http-sqlite.json">SQLite HTTP</a> · <a href="ingestion.json">Ingestion</a> · <a href="import.json">Import</a></p></section>
<script>const data=__DATA__;const engines=['sqlite','rocksdb'];document.getElementById('http').innerHTML='<tr><th>Endpoint</th><th>SQLite p50 / p95</th><th>RocksDB p50 / p95</th></tr>'+['tree','nearby','terms'].map(n=>'<tr><td>/v1/'+n+'</td>'+engines.map(b=>{const t=data.pilots[b].timings[n];return '<td>'+t.p50_ms.toFixed(1)+' / '+t.p95_ms.toFixed(1)+' ms</td>'}).join('')+'</tr>').join('');function render(){const m=document.getElementById('metric').value;const values=engines.map(b=>{const p=data.pilots[b];return m==='ingest'?data.ingestion[b].posts_per_second:(m==='metadata'?p.storage_stats.metadata_bytes:m==='total'?p.storage_stats.total_bytes:p.process_memory.VmHWM)/1048576});const max=Math.max(...values);document.getElementById('bars').innerHTML=engines.map((b,i)=>'<p>'+b+': '+values[i].toFixed(1)+'</p><div class="bar" style="width:'+100*values[i]/max+'%;background:'+['#83a9ef','#67c7b3'][i]+'"></div>').join('')}document.getElementById('metric').onchange=render;render();</script></html>'''
(OUT / 'demo.html').write_text(html.replace('__DATA__', data))
print('Wrote integrated RocksDB report and viewer')
