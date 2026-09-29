"""Generate auditable Python 0.6.0 scoring/tokenization fixtures; not a runtime dependency."""
import json
from pathlib import Path
import yake
from importlib.metadata import version
assert version("yake") == "0.6.0", "fixture requires pinned yake==0.6.0"
from yake.data import DataCore
from yake.data.utils import tokenize_sentences, pre_filter
ROOT=Path(__file__).resolve().parents[2]
texts=[
 'NASA builds rockets. NASA rockets reach space.',
 'Space telescopes observe distant galaxies and stars.',
 'The art of science and the science of art.',
 'Cats cat cats CAT cat. Dogs dog dogs DOG.',
 'I love Julia, Python and C++. Space-time matters.',
 "I don’t like this. I can't say it's bad.",
 'Dr. Smith visits New York. They use U.S. data.',
 '你好世界。こんにちは！ café naïve résumé',
 'Alpha: beta; gamma / delta — epsilon.',
 'one\ntwo\nThree', 'Wow!!! Amazing... Right? yes',
 '15:30 3.14 1,000 COVID-19', '', 'the and or', '!!! 🦋',
 'Climate change is real. Climate change affects people. Climate change matters.',
 'The international space station observes the planet. A station in space.',
 'Quantum physics quantum physics quantum physics.',
 'Julia &amp; Python work. café café CAFÉ.',
]
# Repeatable real-post sample to catch scoring differences beyond hand-built examples.
with (ROOT/'data/gte_disk_ablation/posts.jsonl').open() as f:
 for i,line in enumerate(f):
  if i==500: break
  texts.append(json.loads(line)['text'])
rows=[]
for n,window,dedup in [(3,1,.9),(1,1,1.),(4,2,.8)]:
 engine=yake.KeywordExtractor(lan='en',n=n,window_size=window,top=24,dedup_lim=dedup)
 for text in texts:
  rows.append(dict(text=text,n=n,window=window,dedup=dedup,
     tokens=tokenize_sentences(pre_filter(text.replace('\n',' '))),
     keywords=engine.extract_keywords(text)))
out=ROOT/'julia-yake/test/python_reference.json';out.write_text(json.dumps(rows,ensure_ascii=False)+'\n')
print(f'{len(rows)} Python reference cases -> {out}')
