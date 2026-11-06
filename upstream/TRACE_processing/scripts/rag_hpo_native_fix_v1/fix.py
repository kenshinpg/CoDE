"""Rebuild the SAME frozen cohort/index inputs with native tokenizer configuration."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import retrieval_core as core


def init(a):
    if a.out.exists():
        raise ValueError('Use a fresh output directory; never reuse old vectors/candidates')
    old = core.read(a.source/'manifest.json')
    for name, digest in old['model_files'].items():
        if core.sha(a.model/name) != digest:
            raise ValueError('Model identity mismatch: '+name)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    probes = ['Fever', 'Diarrhea', 'Vomiting', 'Abdominal pain']
    for phrase in probes:
        if tok.unk_token_id in tok(phrase)['input_ids']:
            raise ValueError('Native tokenizer still returns UNK for '+phrase)
    a.out.mkdir(parents=True)
    hashes = {}
    for name in ('terms.json', 'phrases.json', 'mentions.jsonl'):
        shutil.copyfile(a.source/name, a.out/name)
        hashes[name] = core.sha(a.out/name)
    new = dict(old)
    new.update(model=str(a.model.resolve()), protocol='native_tokenizer_mean_pool_v1',
               parent_manifest_sha256=core.sha(a.source/'manifest.json'),
               frozen_inputs_sha256=hashes,
               tokenizer_policy='AutoTokenizer native configuration; no lower-case override',
               tokenizer_do_lower_case=getattr(tok, 'do_lower_case', None),
               code_sha256=core.sha(__file__), retrieval_core_sha256=core.sha(Path(core.__file__)))
    core.write(a.out/'manifest.json', new)
    print('Frozen cohort and HPO terms copied; no resampling. Native tokenizer:',
          getattr(tok, 'do_lower_case', None), flush=True)


def gate(a):
    import numpy as np
    from transformers import AutoTokenizer
    m = core.read(a.run/'manifest.json')
    if m['protocol'] != 'native_tokenizer_mean_pool_v1':
        raise ValueError('Unexpected protocol')
    phrases, terms = core.read(a.run/'phrases.json'), core.read(a.run/'terms.json')
    counts = np.zeros(len(phrases), dtype=np.int32)
    vec = np.zeros((len(phrases), 768), dtype=np.float32)
    for rank in range(8):
        z = np.load(a.run/f'embeddings{rank}.npz', allow_pickle=False)
        if str(z['phrases_sha256']) != core.sha(a.run/'phrases.json'):
            raise ValueError('Stale shard')
        ids = z['ids']
        np.add.at(counts, ids, 1)
        vec[ids] = z['vectors']
    if not (counts == 1).all() or not np.isfinite(vec).all():
        raise ValueError('Invalid shard coverage/vectors')
    tok = AutoTokenizer.from_pretrained(m['model'], local_files_only=True)
    unknown_tokens = 0
    for off in range(0, len(phrases), 512):
        ids = tok([x['text'] for x in phrases[off:off+512]], truncation=False)['input_ids']
        unknown_tokens += sum(row.count(tok.unk_token_id) for row in ids)
    wanted = ['Fever', 'Diarrhea', 'Vomiting', 'Abdominal pain']
    term_ids = {t['name']:t['id'] for t in terms}
    random_terms = sorted(terms, key=lambda t: hashlib.sha256(t['id'].encode()).hexdigest())[:128]
    queries = list(dict.fromkeys(wanted + [t['name'] for t in random_terms]))
    import torch
    torch.set_num_threads(16)
    encoder = core.encoder(m['model'], 'cpu')
    rows = []
    for off in range(0, len(queries), 32):
        q = queries[off:off+32]
        vectors, _ = encoder(q)
        for text, scores in zip(q, vectors@vec.T):
            chosen, seen = [], set()
            for idx in np.argsort(-scores, kind='stable'):
                t = terms[phrases[int(idx)]['term_index']]
                if t['id'] in seen: continue
                seen.add(t['id']); chosen.append({'id':t['id'], 'name':t['name'], 'cosine':float(scores[idx])})
                if len(chosen) == 20: break
            rows.append({'query':text, 'expected':term_ids[text],
                         'expected_in_top20':term_ids[text] in seen, 'top5':chosen[:5]})
    passed = unknown_tokens == 0 and all(r['expected_in_top20'] for r in rows if r['query'] in wanted)
    report = {'gate_passed':passed, 'unknown_tokens_in_hpo_names':unknown_tokens,
              'exact_name_queries':len(rows), 'exact_name_top20_hits':sum(r['expected_in_top20'] for r in rows),
              'queries':rows, 'note':'Engineering identity checks, NOT clinical linking accuracy. Gate requires four fixed probes and zero HPO-name UNK tokens.'}
    core.write(a.run/'native_gate.json', report)
    print({k:v for k,v in report.items() if k != 'queries'}, flush=True)
    if not passed: raise ValueError('Retrieval gate failed; mapping not started')


def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest='cmd', required=True)
    q=s.add_parser('init'); q.add_argument('--source', type=Path, required=True)
    q.add_argument('--model', type=Path, required=True); q.add_argument('--out', type=Path, required=True)
    q=s.add_parser('embed'); q.add_argument('--run', type=Path, required=True); q.add_argument('--batch-size', type=int, default=64)
    q=s.add_parser('gate'); q.add_argument('--run', type=Path, required=True)
    q=s.add_parser('retrieve'); q.add_argument('--run', type=Path, required=True)
    q.add_argument('--batch-size', type=int, default=32); q.add_argument('--cpu-threads', type=int, default=16)
    a=p.parse_args()
    if a.cmd in ('embed','retrieve'):
        if core.read(a.run/'manifest.json').get('protocol') != 'native_tokenizer_mean_pool_v1':
            raise ValueError('Refuse old index')
        if a.cmd == 'retrieve' and not core.read(a.run/'native_gate.json')['gate_passed']:
            raise ValueError('Gate not passed')
        getattr(core, a.cmd)(a)
    else: globals()[a.cmd](a)


if __name__ == '__main__': main()
