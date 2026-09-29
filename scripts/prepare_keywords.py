"""Stream raw extraction sidecars; Julia owns canonicalization and storage policy.
Reusable by embedding preparation; never reads or modifies embedding values.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from keyword_worker import extract_batch, EXTRACTOR

def extract_page(page):
    t=time.perf_counter()
    result=extract_batch([dict(text=p['text'],language=p.get('language','')) for p in page])
    return [dict(identity=p.get('uri',str(p.get('id',''))),text_sha256=hashlib.sha256(p['text'].encode()).hexdigest(),**r) for p,r in zip(page,result['results'])],time.perf_counter()-t

def prepare(source, output, workers=4):
    source,output=Path(source),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    digest=hashlib.file_digest(source.open('rb'),'sha256').hexdigest()
    contract=dict(source_sha256=digest,extractor=EXTRACTOR,workers=workers)
    manifest=output/'manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['contract']==contract and old['sidecar_sha256']==hashlib.file_digest((output/'keywords.jsonl').open('rb'),'sha256').hexdigest():
            return old
        raise RuntimeError('Existing keyword artifact has a different contract')
    start=time.perf_counter();n=0;times=[];truncated=0
    with source.open() as src, (output/'keywords.jsonl.tmp').open('w') as dst, ProcessPoolExecutor(max_workers=workers) as pool:
        # Keep at most two pages per worker in flight.
        while True:
            pages=[]
            for _ in range(2*workers):
                page=[]
                for _ in range(128):
                    line=src.readline()
                    if not line: break
                    page.append(json.loads(line))
                if page: pages.append(page)
            if not pages: break
            for rows,elapsed in pool.map(extract_page,pages):
                times.append(elapsed)
                for row in rows:
                    dst.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
                    n+=1;truncated+=row['truncated']
            if n%8192==0: print(f'{n:,} keyword posts; {n/(time.perf_counter()-start):.1f}/s',flush=True)
    (output/'keywords.jsonl.tmp').replace(output/'keywords.jsonl')
    wall=time.perf_counter()-start
    times.sort()
    result=dict(contract=contract,count=n,wall_seconds=wall,posts_per_second=n/wall,batch_size=128,
                worker_batch_seconds=times,worker_batch_p50=times[len(times)//2],worker_batch_p95=times[min(len(times)-1,int(len(times)*.95))],
                truncated=truncated,sidecar_bytes=(output/'keywords.jsonl').stat().st_size,
                sidecar_sha256=hashlib.file_digest((output/'keywords.jsonl').open('rb'),'sha256').hexdigest(),
                versions={p:importlib.metadata.version(p) for p in ['yake','segtok','jellyfish','numpy','regex','networkx']})
    manifest.write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='data/gte_disk_ablation/posts.jsonl');p.add_argument('--output',default='data/keyword_preparation');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.output,a.workers),indent=2))
