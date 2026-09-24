"""Build chronological row references; retain 2024 timestamps from the 100k archive."""
import datetime
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
rows=[]
for i,line in enumerate((root/'data/bluesky_100000/posts.jsonl').open()):
    row=json.loads(line)
    try:
        t=datetime.datetime.fromisoformat(row['created_at'].replace('Z','+00:00'))
    except (KeyError,ValueError):
        continue
    if t.year==2024 and t.tzinfo is not None:
        rows.append((t.timestamp(),i+1))
rows.sort()
if not rows: raise ValueError('No eligible timestamps')
out=root/'output/leaf_decay';out.mkdir(parents=True,exist_ok=True)
start=rows[0][0]
(out/'chronology.csv').write_text(''.join(f'{i},{t-start}\n' for t,i in rows))
print(f'{len(rows)} chronological posts')
