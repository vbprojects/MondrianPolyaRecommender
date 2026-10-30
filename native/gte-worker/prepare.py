"""One-time Linux bundle preparation. Python is not used by the deployed worker."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REVISION = '4cc5e73d86a67c601897257b467187234aa3bca3'
ASSETS = {
    'onnx/model_quantized.onnx': 'ceab9964bcdb19ad013a2fe44bc1ddb3657ab0af1f48cddc349014d6085762ca',
    'tokenizer.json': 'da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'output/gte_tiny_rust')
    parser.add_argument('--ort-library', type=Path, help='Trusted ONNX Runtime >=1.30 Linux shared library')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hub = Path(os.environ.get('HF_HUB_CACHE', str(Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface')))/'hub')))
    snapshot = hub/'models--TaylorAI--gte-tiny'/'snapshots'/REVISION
    for name, checksum in ASSETS.items():
        target = args.output/Path(name).name
        if target.exists() and sha(target) == checksum:
            continue
        temporary = target.with_suffix(target.suffix+'.partial')
        try:
            if (snapshot/name).exists():
                shutil.copyfile(snapshot/name, temporary)
            else:
                url = f'https://huggingface.co/TaylorAI/gte-tiny/resolve/{REVISION}/{name}'
                with urllib.request.urlopen(url, timeout=60) as src, temporary.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
            if sha(temporary) != checksum:
                raise ValueError(f'Pinned asset checksum mismatch: {name}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    library = args.ort_library
    if library is None:
        spec = importlib.util.find_spec('onnxruntime')
        if spec is None:
            parser.error('Pass --ort-library or run preparation in the existing ONNX Python environment')
        candidates = sorted((Path(spec.origin).parent/'capi').glob('libonnxruntime.so.*'))
        if len(candidates) != 1:
            parser.error('Cannot select a native runtime; pass --ort-library')
        library = candidates[0]
    destination = args.output/'libonnxruntime.so'
    if library.resolve() != destination.resolve():
        shutil.copyfile(library, destination)
    metadata = dict(revision=REVISION, files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
        for p in [args.output/'model_quantized.onnx', args.output/'tokenizer.json', destination]})
    (args.output/'bundle.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Prepared {args.output}; native runtime copied without Python bindings or GPU providers.')


if __name__ == '__main__':
    main()
