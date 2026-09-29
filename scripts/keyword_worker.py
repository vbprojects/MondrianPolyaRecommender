"""YAKE adapter shared by preparation and the keyword-only worker; no ML imports."""
import importlib.metadata
import json
import re
import sys
from functools import lru_cache

EXTRACTOR = 'yake@0.6.0;en;n=3;dedup=0.9;window=1;top=24;maxchars=8192;clean-v1'

@lru_cache(maxsize=1)
def engine():
    import yake
    if importlib.metadata.version('yake') != '0.6.0':
        raise RuntimeError('Expected pinned yake==0.6.0')
    return yake.KeywordExtractor(lan='en', n=3, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=24)

def extract_batch(inputs):
    if not 0 <= len(inputs) <= 1024:
        raise ValueError('batch exceeds 1024')
    results = []
    for item in inputs:
        text = item['text']
        lang = (item.get('language') or '').lower()
        truncated = len(text) > 8192
        text = text[:8192]
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'(?<!\w)@[\w.:-]+(?:@[\w.-]+)?', ' ', text)
        text = text.replace('#', '')
        pairs = engine().extract_keywords(text) if (not lang or lang.replace('_','-').split('-')[0]=='en') and text.strip() else []
        results.append(dict(candidates=[dict(phrase=p, rank=i+1, score=float(s)) for i,(p,s) in enumerate(pairs)], truncated=truncated))
    return dict(extractor=EXTRACTOR, results=results)

def main():
    engine()
    print(json.dumps(dict(extractor=EXTRACTOR)), flush=True)
    for line in sys.stdin:
        try:
            if len(line.encode()) > 80*1024*1024:
                raise ValueError('request too large')
            r=json.loads(line)
            if r['op'] != 'keywords':
                raise ValueError('unknown operation')
            out=extract_batch(r['inputs'])
        except Exception as exc:
            out=dict(error=type(exc).__name__)  # no input text in diagnostics
        print(json.dumps(out,ensure_ascii=False,separators=(',',':')),flush=True)
if __name__ == '__main__':
    main()
