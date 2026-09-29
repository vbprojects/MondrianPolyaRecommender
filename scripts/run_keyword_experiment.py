"""Run the keyword experiment sequentially on fresh stores; no live-index changes."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
if any((ROOT/'data/keyword_experiment'/m).exists() for m in ['disabled','hashtags','yake','online']):
    raise SystemExit('Experiment stores already exist. Preserve or move them before a fresh run.')
python=ROOT/'.venv-embeddings/bin/python'
def run(*args):
    subprocess.run([str(x) for x in args],cwd=ROOT,check=True)
run(python,'scripts/prepare_keywords.py','--workers','4')
for suite in ['disk_keywords.jl','keyword_crash_process.jl','disk_tree_store.jl']:
    run('julia','--project=environments/storage',f'test/{suite}')
for method in ['disabled','hashtags','yake','online']:
    run('julia','--project=environments/storage','examples/keyword_experiment.jl',method)
run('julia','--project=environments/storage','examples/validate_keywords.jl')
run(sys.executable,'scripts/keyword_report.py')
run('node','test/keyword_viewer.cjs')
