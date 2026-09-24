"""Offline report of measured rank-4 tree growth."""
import json, html
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'output/bluesky_million'
r=json.loads((out/'results.json').read_text());e=json.loads((root/'data/bluesky_1000000/manifest.json').read_text());f=r['final'];mib=2**20
summary=f'''One million Bluesky posts — leaf-local online rank-4 PPCA
Posts: {r['count']:,}; dimensions: {r['dimensions']}; static, no decay
Cuts: {f['cuts']:,}; leaves: {f['leaves']:,}; nodes: {f['nodes']:,}; maximum depth: {f['max_depth']}
Tree retained memory: {f['tree_bytes']/mib:.2f} MiB (includes reservoirs, bounds, PPCA states)
PPCA state containers: {f['states_bytes']/mib:.2f} MiB; reservoirs: {f['reservoir_bytes']/mib:.2f} MiB
Process RSS at last checkpoint: {f['rss_bytes']/mib:.2f} MiB; peak RSS: {f['peak_rss_bytes']/mib:.2f} MiB
Full embedding archive: {r['embedding_bytes']/mib:.2f} MiB; serialized model: {r['model_file_bytes']/mib:.2f} MiB
Fit including tracking: {r['fit_seconds']:.2f} seconds
PPCA updates: {r['update_seconds']:.2f}s; moment-transfer splits: {r['split_seconds']:.2f}s
GPU embedding inference: {e['batch_encoding_seconds']:.2f}s; tokenization: {e['tokenization_seconds']:.2f}s
Transferred leaf mass versus exact post routing: TV={r['train_leaf_mass_tv']:.6f}
Total preserved mass: {r['mass']:.6f}
'''
(out/'summary.txt').write_text(summary)
def chart(key,title,div=1):
    data=r['tracking'];top=max(p[key]/div for p in data)*1.05 or 1
    pts=' '.join(f"{55+850*p['posts']/r['count']:.1f},{250-210*p[key]/div/top:.1f}" for p in data)
    return f'<h2>{title}</h2><svg viewBox="0 0 940 290"><path d="M55 30V250H910" fill="none" stroke="#888"/><polyline points="{pts}" fill="none" stroke="#58b8fa" stroke-width="3"/><text x="5" y="35">{top:.1f}</text><text x="30" y="255">0</text><text x="55" y="277">0 posts</text><text x="790" y="277">1,000,000 posts</text></svg>'
body='<h1>One million posts: rank-4 PPCA tree</h1><pre>'+html.escape(summary)+'</pre>'
for key,title,div in [('leaves','Active leaves',1),('tree_bytes','Retained tree memory (MiB)',mib),('rss_bytes','Process resident memory (MiB)',mib),('max_depth','Maximum tree depth',1)]:body+=chart(key,title,div)
body+='''<h2>Interpretation and limits</h2><p>One Gaussian per active leaf. At a cut the parent is replaced by two conditional-moment approximations. Internal nodes retain routing geometry but no PPCA model or reservoir. Rank is four, reservoir capacity is 64 per leaf, and splitting requires 512 arrivals since leaf creation. Limits: 10,000 leaves and depth 24. The cut axis uses reservoir coordinate ranges; thresholds lie between its quartiles. This is a data-adaptive Mondrian-style heuristic, not an exact Mondrian process.</p><p>Tree memory is Julia reachable-object size. It excludes the input archive, runtime, allocation garbage, tracking records and per-post IDs/codes. Process RSS includes the memory-mapped input pages and Julia runtime. These are different measurements, not alternative estimates of the same storage. State and reservoir breakdowns have small separate container overheads.</p><p>One million unique-URI, nonblank posts sampled from the first 60,000 records in each of 20 pinned dataset shards, then ingested in seed-192 random order. This is not a chronological day or a corpus-wide uniform sample. GPU computes embeddings; the tree fits on CPU. Gaussian moment transfer ignores inherited box truncation; retained mass is conserved but historical leaf assignment is approximate. No recommendation quality claim follows from this scaling run.</p><p><a href="memory.csv">Memory checkpoints</a> · <a href="cuts.csv">Every cut</a> · <a href="results.json">Results JSON</a> · <a href="summary.txt">Summary</a></p>'''
(out/'demo.html').write_text('<!doctype html><meta charset="utf-8"><title>Million-post PPCA tree</title><style>body{max-width:1000px;margin:40px auto;background:#111827;color:#e5e7eb;font:16px system-ui;padding:20px}pre{white-space:pre-wrap;line-height:1.7}svg{width:100%;background:#182235}text{fill:#ddd;font-size:13px}a{color:#58b8fa}</style>'+body)
print(summary)
