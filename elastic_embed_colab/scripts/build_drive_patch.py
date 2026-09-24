"""Embed current diagnostic fixes in notebooks 01, 02, and 03 for execution on its Colab kernel.

Run locally after changing these files. The generated cell backs up Drive files
and does not depend on the updated source already being uploaded to Drive.
"""
import ast
import base64
import json
from pathlib import Path
import zlib


root = Path(__file__).resolve().parents[1]
paths = ["src/elastic_embed/sanity.py", "scripts/sanity_check.py", "src/elastic_embed/initialization.py", "src/elastic_embed/model.py", "src/elastic_embed/runtime.py", "scripts/bootstrap_colab.sh"]
payload = base64.b64encode(zlib.compress(json.dumps({
    path: (root / path).read_text() for path in paths
}).encode())).decode()
for notebook_name in ("01_teacher_cache.ipynb", "02_initialization_recovery.ipynb", "03_train.ipynb"):
    notebook_path = root / "notebooks" / notebook_name
    notebook = json.loads(notebook_path.read_text())
    notebook["cells"] = [cell for cell in notebook["cells"]
                         if not cell.get("metadata", {}).get("elastic_embed_drive_patch")]
    config_index = next(i for i, cell in enumerate(notebook["cells"])
                        if "def run(command, *args):" in "".join(cell["source"]))
    runner = "".join(notebook["cells"][config_index]["source"]).split("def run(command, *args):", 1)[1]
    source = f'''# Apply the bundled runner patch to the connected Colab Drive.
import base64, json, subprocess, time, zlib
from pathlib import Path

BUNDLE = Path(globals().get('BUNDLE', '/content/drive/MyDrive/elastic_embed_colab'))
assert (BUNDLE / 'pyproject.toml').is_file(), f'Check the mounted bundle path: {{BUNDLE}}'
_patch_files = json.loads(zlib.decompress(base64.b64decode({payload!r})))
_patch_stamp = str(time.time_ns())
for _relative, _source in _patch_files.items():
    _target = BUNDLE / _relative
    _target.parent.mkdir(parents=True, exist_ok=True)
    if _target.exists() and _target.read_text() == _source:
        print(f'Already current: {{_relative}}')
        continue
    _backup = _target.with_name(_target.name + '.before-' + _patch_stamp)
    if _target.exists():
        _backup.write_bytes(_target.read_bytes())
    _temporary = _target.with_name(_target.name + '.update-' + _patch_stamp)
    _temporary.write_text(_source)
    _temporary.replace(_target)
    print(f'Updated {{_relative}}' + (f'; backup: {{_backup.name}}' if _backup.exists() else ' (new file)'))

def run(command, *args):{runner}
print('Patch applied. Rerun the failed command; detailed failures will be printed and logged.')
'''
    ast.parse(source)
    notebook["cells"][config_index + 1:config_index + 1] = [
        {"cell_type": "markdown", "metadata": {"elastic_embed_drive_patch": True},
         "source": ["### Update the Drive copy from this notebook\n",
                    "Run the next cell to apply the bundled runner fixes and independent sanity-check script directly through this notebook’s Colab kernel. It backs up replaced source files and updates the output/logging helper. Configs, caches, and checkpoints are unchanged. This cell works even if only this notebook was updated locally.\n"]},
        {"cell_type": "code", "execution_count": None, "outputs": [],
         "metadata": {"elastic_embed_drive_patch": True}, "source": source.splitlines(keepends=True)},
    ]
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"Embedded {len(paths)} source files in {notebook_path}")
