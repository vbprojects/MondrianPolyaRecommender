"""Render measured keyword comparison, without inventing semantic quality scores."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'output/keywords'
methods={m:json.loads((out/f'{m}.json').read_text()) for m in ['disabled','hashtags','yake']}
extraction=json.loads((ROOT/'data/keyword_preparation/manifest.json').read_text())
base=sum(methods['disabled']['db_pages'].values())
extra={m:sum(r['db_pages'].values())-base for m,r in methods.items()}
physical_extra={m:r["stats"]["total_bytes"]-methods["disabled"]["stats"]["total_bytes"] for m,r in methods.items()}
equal=all(r['tree_signature']==methods['disabled']['tree_signature'] for r in methods.values())
assert equal,'Keyword tracking changed tree cuts'
rate=methods['yake']['storage_posts_per_second']
serial=100000/(extraction['wall_seconds']+methods['yake']['ingest_with_prepared_keywords_seconds'])
recommendation=(f'Keep the pluggable YAKE adapter as an optional enrichment step. This run sustained {rate:.0f} storage posts/s; '
                f'a serial composition of measured preparation and ingestion stages is {serial:.0f} posts/s, excluding embedding inference. '
                f'Exact keyword tables add {extra["yake"]/100000:.0f} database bytes per post here. '
                'Review the displayed labels before using them as client interest suggestions. This is one seed and one 100,000-post corpus, not weekly-scale validation.')
recommendation+=f' The 128-to-5 merge took {methods["yake"]["merge_seconds"]:.0f}s with YAKE versus {methods["disabled"]["merge_seconds"]:.0f}s without keywords; bounded batching of count updates is the main maintenance optimization.'
online=json.loads((out/'online.json').read_text()) if (out/'online.json').exists() else None
if online:
    recommendation+=f' A separate 10,000-post run through the actual single-worker YAKE adapter and durable ingestion achieved {online["end_to_end_posts_per_second"]:.0f} posts/s (embedding inference excluded).'
data=dict(methods=methods,online=online,extraction=extraction,extra_db_bytes=extra,extra_physical_bytes=physical_extra,routing_equal=equal,recommendation=recommendation,
          serial_composed_posts_per_second=serial)
(out/'results.json').write_text(json.dumps(data,indent=2)+'\n')
(out/'config.json').write_text(json.dumps(dict(tree=methods['yake']['config'],keyword=methods['yake']['keyword_config'],extraction=extraction['contract']),indent=2)+'\n')
html=(ROOT/'examples/keyword_viewer.html').read_text().replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c'))
(out/'demo.html').write_text(html)
lines=['Keyword experiment — 100,000 real posts',f'Extraction: {extraction["posts_per_second"]:.1f} posts/s, {extraction["wall_seconds"]:.2f}s, 4 CPU workers',f'Identical routing across all methods: {equal}']
for m,r in methods.items():
    lines.append(f'{m}: {r["storage_posts_per_second"]:.1f} storage posts/s; {extra[m]/1048576:.2f} added DB MiB; {r["memory"]["rss"]/1048576:.1f} RSS MiB; merge {r["merge_seconds"]:.3f}s')
lines.extend([f'YAKE extra measured files including WAL: {physical_extra["yake"]/1048576:.2f} MiB','',recommendation])
(out/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
