"""Small authored semantic diagnostics; no teacher weights, targets, or external dataset."""
import argparse
import contextlib
import json
from pathlib import Path
import time
import traceback

import torch
from torch.nn import functional as F
from .config import load_config, identity
from .data import tokenizer
from .io import atomic_json, checksum
from .training import load_student

# query, relevant passage, same-topic hard negative, unrelated negative.
# Deliberately diagnostic examples, not an official or statistically powered benchmark.
CASES = [
    ('cancel', 'How do I cancel my subscription?', 'To end your paid membership, open settings and select cancellation.', 'To renew your paid membership, open settings and select renewal.', 'The bakery sells fresh bread each morning.'),
    ('refund', 'Can I get my money back for a broken item?', 'Faulty purchases qualify for a refund to the original payment method.', 'Faulty purchases can be replaced, but refunds are not offered.', 'The train leaves from platform seven.'),
    ('password', 'I forgot my password. How can I sign in?', 'Use the password recovery link to regain access to your account.', 'You can change your profile picture after signing in.', 'Rain is expected throughout the afternoon.'),
    ('delivery', 'Where is my parcel? It still has not arrived.', 'Track your missing shipment using the tracking number in your order email.', 'Your parcel was delivered successfully yesterday.', 'This soup needs a little more salt.'),
    ('battery', 'My phone dies after just an hour.', 'The handset battery drains rapidly and needs frequent charging.', 'The handset battery lasts for several days between charges.', 'The football match ended in a draw.'),
    ('wifi', 'My laptop cannot connect to the internet.', 'Troubleshoot the computer network connection and restart the wireless router.', 'The computer is connected and browsing websites normally.', 'The museum displays ancient pottery.'),
    ('sleep', 'I keep waking up in the middle of the night.', 'My sleep is repeatedly interrupted and I cannot stay asleep.', 'I sleep soundly through the entire night.', 'The mechanic changed the brake pads.'),
    ('noise', 'The neighbors are keeping me awake with loud music.', 'Music from the apartment next door is disturbing my sleep.', 'The neighbors have stopped playing music and it is quiet.', 'I planted tomatoes in the garden.'),
    ('lostcat', 'Has anyone seen my missing cat?', 'Our pet feline disappeared; please contact us if you spot her.', 'Our missing cat has been found and is safely home.', 'A new software update is available.'),
    ('bus', 'Is the bus running late?', 'The bus service is delayed and will arrive behind schedule.', 'The bus service is running on time today.', 'These shoes are available in blue.'),
    ('job', 'I am looking for my first job.', 'Seeking entry-level employment with no previous work experience.', 'We require applicants to have ten years of professional experience.', 'The river flooded after heavy rainfall.'),
    ('rent', 'My landlord is raising the rent again.', 'The property owner has announced another increase in monthly housing payments.', 'The property owner has reduced the monthly rent.', 'The concert starts at eight tonight.'),
    ('allergy', 'I need food that does not contain peanuts.', 'Please provide a peanut-free meal because of a food allergy.', 'This meal contains peanuts and peanut oil.', 'The printer has run out of paper.'),
    ('backup', 'How can I protect my photos if my phone breaks?', 'Keep a backup copy of your pictures on another device or storage service.', 'Delete all backup copies of your pictures from other devices.', 'The cafe closes early on Sundays.'),
    ('meet', 'Can we move our meeting to tomorrow?', 'Please reschedule our appointment for the following day.', 'Our appointment must remain today and cannot be moved.', 'The kitten is playing with a toy.'),
    ('slang', 'ngl this app keeps crashing lol', 'Honestly, the application repeatedly closes unexpectedly.', 'The application is stable and never closes unexpectedly.', 'The restaurant serves excellent noodles.'),
    ('tickets', 'Anyone selling a spare ticket for tonight?', 'I want to buy an extra admission pass for this evening.', 'I have an extra admission pass that I want to sell.', 'The mountain trail is covered in snow.'),
    ('weather', 'Looks like the storm knocked out our power.', 'The electricity supply failed during severe weather.', 'Electricity remained available throughout the storm.', 'The library has a new poetry section.'),
    ('privacy', 'I do not want strangers seeing my posts.', 'Restrict your social profile so only approved followers can view updates.', 'Make your social profile public so everyone can view updates.', 'The oven takes ten minutes to heat up.'),
    ('soldout', 'Are there any seats left for the show?', 'Check whether admission tickets are still available for the performance.', 'All admission tickets for the performance have sold out.', 'The phone camera has three lenses.'),
]
# Cross-language spot checks use the same English candidates as the English queries.
TRANSLATIONS = [
    ('es', 'cancel', '¿Cómo puedo cancelar mi suscripción?'),
    ('fr', 'password', "J’ai oublié mon mot de passe. Comment me connecter ?"),
    ('de', 'delivery', 'Wo ist mein Paket? Es ist noch nicht angekommen.'),
    ('pt', 'battery', 'A bateria do meu celular acaba em apenas uma hora.'),
    ('hi', 'lostcat', 'क्या किसी ने मेरी खोई हुई बिल्ली को देखा है?'),
    ('ar', 'wifi', 'لا يستطيع حاسوبي المحمول الاتصال بالإنترنت.'),
    ('ja', 'bus', 'バスは遅れていますか？'),
    ('zh', 'refund', '买到损坏的商品可以退款吗？'),
]


def fixture():
    texts, indices = [], {}
    def add(text):
        if text not in indices:
            indices[text] = len(texts)
            texts.append(text)
        return indices[text]
    cases = [dict(id=key, query=add(q), positive=add(p), hard=add(h), easy=add(e))
             for key, q, p, h, e in CASES]
    by_id = {row['id']: row for row in cases}
    translations = [dict(language=lang, query=add(q), positive=by_id[key]['positive'])
                    for lang, key, q in TRANSLATIONS]
    candidates = list(dict.fromkeys(row[k] for row in cases for k in ('positive', 'hard', 'easy')))
    return texts, cases, translations, candidates


def retrieval(vectors, queries, positives, candidates):
    scores = vectors[queries] @ vectors[candidates].T
    target = torch.tensor([candidates.index(p) for p in positives])
    correct = scores[torch.arange(len(queries)), target]
    # Conservative tie handling: a collapsed model must not win by index order.
    ranks = (scores >= correct[:, None] - 1e-7).sum(1)
    return {'queries': len(queries), 'candidates': len(candidates),
            'hit@1': float((ranks <= 1).float().mean()),
            'hit@5': float((ranks <= 5).float().mean()),
            'mrr': float((1 / ranks.float()).mean()),
            'random_hit@1': 1 / len(candidates), 'random_hit@5': min(5, len(candidates)) / len(candidates)}


def score(raw, cases, translations, candidates):
    norms = raw.norm(dim=-1)
    if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(norms).all()):
        return {'status': 'nonfinite', 'flags': ['Nonfinite embeddings: investigate before further training.']}
    vectors = F.normalize(raw.float(), dim=-1)
    gram = vectors @ vectors.T
    off = gram[~torch.eye(len(raw), dtype=torch.bool)]
    spread = float((vectors - vectors.mean(0)).square().sum(-1).mean())
    flags = []
    if bool((norms < 1e-8).any()):
        flags.append('Near-zero output vectors.')
    if spread < 1e-4:
        flags.append('Nearly constant normalized vectors: possible collapse.')
    elif float(off.mean()) > .95:
        flags.append('Very high average similarity across distinct probe texts; inspect anisotropy.')
    details = []
    for row in cases:
        p, h, e = [float(gram[row['query'], row[k]]) for k in ('positive', 'hard', 'easy')]
        details.append({'id': row['id'], 'positive_cosine': p, 'hard_cosine': h, 'easy_cosine': e,
                        'easy_margin': p - e, 'hard_margin': p - h,
                        'easy_correct': p > e + 1e-6, 'hard_correct': p > h + 1e-6})
    mean = lambda key: sum(row[key] for row in details) / len(details)
    if mean('easy_correct') <= .5:
        flags.append('Easy semantic ranking is at or below 50%; inspect individual examples.')
    queries, positives = [r['query'] for r in cases], [r['positive'] for r in cases]
    return {'status': 'ok', 'flags': flags,
            'raw_norm_min': float(norms.min()), 'raw_norm_mean': float(norms.mean()),
            'raw_norm_max': float(norms.max()), 'normalized_spread': spread,
            'off_diagonal_cosine_mean': float(off.mean()), 'off_diagonal_cosine_std': float(off.std()),
            'easy_accuracy': mean('easy_correct'), 'hard_accuracy': mean('hard_correct'),
            'mean_easy_margin': mean('easy_margin'), 'mean_hard_margin': mean('hard_margin'),
            'retrieval': retrieval(vectors, queries, positives, candidates),
            'cross_language_retrieval': retrieval(vectors, [r['query'] for r in translations],
                                                  [r['positive'] for r in translations], candidates),
            'cross_language_pairs': [dict(language=r['language'], cosine=float(gram[r['query'], r['positive']]))
                                     for r in translations],
            'examples': details}


def evaluate_checkpoint(cfg, path, texts, device, backend, batch_size):
    before = path.stat()
    model, state = load_student(cfg, path, backend)
    saved_teacher = state.get('config', {}).get('teacher', {})
    for key in ('name', 'revision'):
        if key in saved_teacher and saved_teacher[key] != cfg['teacher'][key]:
            raise ValueError('Checkpoint tokenizer identity differs from selected config')
    if model.cfg['vocab_size'] != cfg['model']['vocab_size']:
        raise ValueError('Checkpoint vocabulary differs from selected config')
    tok = tokenizer(cfg)  # Only tokenizer files; never loads the teacher model.
    inputs = tok(texts, padding=True, truncation=False, return_tensors='pt')
    if inputs['input_ids'].shape[1] > 256:
        raise ValueError('Sanity fixture exceeds 256 tokens; refusing silent truncation')
    model.to(device).eval()
    precision = cfg['training']['precision'] if device.startswith('cuda') else 'float32'
    if precision not in ('float32', 'bfloat16'):
        raise ValueError('Unsupported precision')
    outputs = {d: [] for d in model.cfg['depths']}
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            ctx = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bfloat16' else contextlib.nullcontext()
            with ctx:
                result = model(inputs['input_ids'][i:i+batch_size].to(device),
                               inputs['attention_mask'][i:i+batch_size].to(device))
            for depth, value in result.items():
                outputs[depth].append(value.float().cpu())
    elapsed = time.perf_counter() - start
    meta = {'checkpoint': str(path.resolve()), 'sha256': checksum(path), 'stage': state.get('stage', 'initialization'),
            'step': state.get('step'), 'total_steps': state.get('total_steps'),
            'complete': state.get('step', 0) >= state.get('total_steps', 0),
            'device': device, 'backend': backend, 'precision': precision, 'encoding_seconds': elapsed,
            'note': 'MOHAWK freezes the sentence readout and fits blocks independently. Its sentence scores are a baseline, not a block-alignment test.' if path.stem.startswith('mohawk') else None}
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError('Checkpoint changed during evaluation; rerun after that training stage finishes')
    return meta, {depth: torch.cat(values) for depth, values in outputs.items()}, model.cfg['dims']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stages', nargs='+', choices=['initialized', 'mohawk-matrix', 'mohawk-block', 'recovery', 'full-depth', 'prefix', 'matryoshka'],
                        default=['mohawk-block', 'full-depth', 'matryoshka'])
    parser.add_argument('--device', default='auto')
    parser.add_argument('--backend', choices=['auto', 'reference', 'fla'], default='auto')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    cfg = load_config(args.config)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    backend = ('reference' if device == 'cpu' else cfg['model']['backend']) if args.backend == 'auto' else args.backend
    texts, cases, translations, candidates = fixture()
    report = {'fixture_id': identity([CASES, TRANSLATIONS]), 'texts': len(texts),
              'cases': CASES, 'translations': TRANSLATIONS, 'checkpoints': {},
              'limitations': ['Authored, small semantic probes; not an official retrieval benchmark.',
                             'Hard negatives can be semantically similar; low hard accuracy alone does not mean collapse.',
                             'No teacher targets or model weights used; only the pinned tokenizer.',
                             'No guarantee of absence from pretraining or your corpus; do not train on this fixture.',
                             'Off-diagonal statistics include related texts as well as unrelated texts.',
                             'Ties are scored conservatively. Multilingual sample is only one query per language.',
                             'Encoding timing includes all exits and is not an early-exit latency benchmark.']}
    output = Path(args.output) if args.output else Path(cfg['run_dir']) / ('sanity-' + time.strftime('%Y%m%d-%H%M%S') + '.json')
    errors, evaluated = 0, 0
    print('stage              depth/dim   easy   hard   hit@1  hit@5   spread', flush=True)
    for stage in args.stages:
        path = Path(cfg['run_dir']) / (stage + '.pt')
        if not path.exists():
            report['checkpoints'][stage] = {'status': 'missing', 'checkpoint': str(path)}
            print(f'{stage}: checkpoint absent; skipped', flush=True)
            continue
        try:
            meta, vectors, dims = evaluate_checkpoint(cfg, path, texts, device, backend, args.batch_size)
            meta['status'], meta['grid'] = 'ok', {}
            if meta['note']:
                print(meta['note'], flush=True)
            if not meta['complete']:
                print(f'WARNING: {stage} is an incomplete checkpoint', flush=True)
            for depth, raw in vectors.items():
                for dim in dims:
                    metrics = score(raw[:, :dim], cases, translations, candidates)
                    meta['grid'][f'{depth}/{dim}'] = metrics
                    if metrics['status'] != 'ok':
                        print(f'{stage} {depth}/{dim}: {metrics["status"]}', flush=True)
                        continue
                    if (depth, dim) in [(1,64), (1,384), (3,128), (6,192), (12,64), (12,384)]:
                        print(f'{stage:18} {depth:2}/{dim:<3}     {metrics["easy_accuracy"]:.0%}   {metrics["hard_accuracy"]:.0%}'
                              f'   {metrics["retrieval"]["hit@1"]:.0%}   {metrics["retrieval"]["hit@5"]:.0%}   {metrics["normalized_spread"]:.4f}', flush=True)
                    for flag in metrics['flags']:
                        print(f'  {stage} {depth}/{dim}: {flag}', flush=True)
            report['checkpoints'][stage] = meta
            if any(m['status'] != 'ok' for m in meta['grid'].values()):
                meta['status'] = 'nonfinite'
                errors += 1
            evaluated += 1
        except Exception:
            errors += 1
            report['checkpoints'][stage] = {'status': 'error', 'checkpoint': str(path), 'error': traceback.format_exc()}
            print(report['checkpoints'][stage]['error'], flush=True)
        finally:
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            atomic_json(output, report)
    atomic_json(output, report)
    print(f'Full per-dimension metrics and individual examples: {output}', flush=True)
    if errors or not evaluated:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
