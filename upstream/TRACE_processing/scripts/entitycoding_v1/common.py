"""CPU-only provenance helpers; offsets use Python Unicode character indices."""
import hashlib, json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def alignment(raw, clean):
    r=[(i,c) for i,c in enumerate(raw) if not c.isspace()]
    c=[(i,x) for i,x in enumerate(clean) if not x.isspace()]
    if ''.join(x for _,x in r)!=''.join(x for _,x in c):
        raise ValueError('Cleaning changed non-whitespace content; raw offsets not guessed')
    return {ci:ri for (ci,_),(ri,_) in zip(c,r)}

def raw_span(mapping,start,end):
    positions=[mapping[i] for i in range(start,end) if i in mapping]
    if not positions: raise ValueError('Empty/whitespace-only entity')
    return positions[0],positions[-1]+1


def span_issue(text,start,end):
    if not (0 <= start <= end <= len(text)):
        raise ValueError('NER offsets outside source chunk')
    if start==end:return 'ZERO_WIDTH'
    if not text[start:end].strip():return 'WHITESPACE_ONLY'
    return None
