"""Resumable local generation and the original whole-text fact matching protocol."""
import concurrent.futures as cf
from copy import deepcopy
from datetime import datetime, timezone
import ipaddress
import importlib.metadata
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter, deque
from pathlib import Path
from .common import read, rows, dump, digest, index, locked, sha
from . import original_generation as gen, base, segment

class InfrastructureError(RuntimeError):
    def __init__(self, message, http_status=None, error_type=None):
        super().__init__(message)
        self.http_status=http_status; self.error_type=error_type or type(self).__name__

def api(url, body=None, timeout=240):
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('OPENAI_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['OPENAI_API_KEY']
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        # Do not expose echoed clinical request bodies in console diagnostics.
        raise InfrastructureError('HTTP ' + str(e.code) + ' at local inference service', e.code, 'HTTPError') from e
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise InfrastructureError(type(e).__name__ + ' at local inference service', error_type=type(e).__name__) from e
    except json.JSONDecodeError as e:
        raise InfrastructureError('Invalid JSON response at local inference service', error_type='JSONDecodeError') from e

def service(endpoint, model, model_dir, out):
    u = urllib.parse.urlparse(endpoint)
    if u.scheme not in ('http', 'https') or not u.hostname or u.username or u.password or u.query:
        raise ValueError('Use an ordinary local OpenAI-compatible endpoint without credentials in the URL')
    addresses = {x[4][0] for x in socket.getaddrinfo(u.hostname, u.port or 80)}
    if not addresses or any(not ipaddress.ip_address(x).is_private for x in addresses):
        raise ValueError('TRACE requests require the authorized local/private service')
    if not endpoint.endswith('/v1/chat/completions'):
        raise ValueError('Endpoint must end /v1/chat/completions')
    cards = [c for c in api(endpoint.removesuffix('/chat/completions') + '/models')['data'] if c['id'] == model]
    if len(cards) != 1 or not isinstance(cards[0].get('max_model_len'), int):
        raise ValueError('Model name/context unavailable in /v1/models; inspect endpoint/model arguments')
    model_dir = Path(model_dir)
    if not (model_dir / 'config.json').is_file():
        raise FileNotFoundError('Checkpoint metadata missing; set --model-dir: ' + str(model_dir))
    config = read(model_dir / 'config.json')
    weights = sorted(model_dir.glob('*.safetensors'))
    if not weights:
        raise ValueError('No local safetensors checkpoint shards')
    identity = dict(model=model, endpoint=endpoint, model_dir=str(model_dir.resolve()),
                    service={k: cards[0].get(k) for k in ['id', 'root', 'max_model_len']},
                    metadata={p.name: sha(p) for p in model_dir.iterdir()
                              if p.is_file() and (p.suffix in {'.json', '.jinja', '.model'} or p.name in {'.mv', '.msc'})},
                    weights=[dict(name=p.name, bytes=p.stat().st_size) for p in weights],
                    config_dtype=config.get('dtype', config.get('torch_dtype')),
                    quantization_config=config.get('quantization_config'),
                    client_runtime=dict(python=sys.version,pydantic=importlib.metadata.version('pydantic')),
                    identity_scope='Local metadata and shard sizes; deployment weight identity confirmed by operator; API is not a weight attestation.')
    locked(out / 'service_binding.json', identity)
    return cards[0]['max_model_len']

class Client:
    def __init__(self, endpoint, model, limit, out):
        self.endpoint, self.model, self.limit, self.out = endpoint, model, limit, out
        self.token_url = endpoint.removesuffix('/v1/chat/completions') + '/tokenize'

    def traced_call(self, stage, task, mode, attempt, operation, payload, seed, function):
        call_id=uuid.uuid4().hex
        meta=dict(call_id=call_id,stage=stage,task_id=task['task_id'],group_id=task.get('id'),mode=mode,
                  attempt=attempt,operation=operation,time_utc=datetime.now(timezone.utc).isoformat(),seed=seed)
        trace=self.out/'requests_PRIVATE'/stage/(call_id+'.json')
        private=dict(meta,request_parameters=deepcopy(payload),status='started',response=None)
        dump(trace,private)
        try:
            result=function()
        except InfrastructureError as e:
            event=dict(meta,http_status=e.http_status,error_type=e.error_type,
                       request_record=trace.relative_to(self.out).as_posix(),
                       recovery_policy='stop_node_then_resume_same_task; returned-response budget unchanged')
            private.update(status='infrastructure_error',error_type=e.error_type,http_status=e.http_status)
            dump(trace,private);dump(self.out/'infrastructure_events_PRIVATE'/(call_id+'.json'),event)
            raise
        except Exception as e:
            private.update(status='unexpected_error',error_type=type(e).__name__);dump(trace,private)
            raise
        private.update(status='returned',response=result,completed_at_utc=datetime.now(timezone.utc).isoformat())
        dump(trace,private)
        return result

    def traced_tokens(self, stage, task, mode, attempt, messages, budget, seed=None):
        payload=dict(model=self.model,messages=messages,add_generation_prompt=True,
                     chat_template_kwargs={'enable_thinking':False})
        return self.traced_call(stage,task,mode,attempt,'tokenize',payload,seed,lambda:self.tokens(messages,budget))

    def tokens(self, messages, budget):
        n = api(self.token_url, dict(model=self.model, messages=messages, add_generation_prompt=True,
                                    chat_template_kwargs={'enable_thinking': False}), 60)['count']
        if n + budget > self.limit:
            raise InfrastructureError('CONTEXT_OVERFLOW; no patient text was truncated')
        return n

    def request(self, payload):
        return api(self.endpoint, payload)

    def generate(self, task):
        journal = self.out / 'generation_attempts_PRIVATE' / (task['task_id'] + '.json')
        attempts = read(journal) if journal.exists() else []; parsed = None
        next_i=min(len(attempts),2)
        next_seed=int(digest([261013,task['task_id'],'initial',next_i])[:8],16)%2147483647
        count = self.traced_tokens('generation',task,'generation',next_i+1,task['messages'],2048,next_seed)
        if attempts and attempts[-1].get('status') == 'complete':
            parsed = gen.parse_response(attempts[-1]['raw_response'], count)
        for i in range(len(attempts), 3) if parsed is None else []:
            # Original seed rule, applied to the new immutable task identity.
            seed = int(digest([261013, task['task_id'], 'initial', i])[:8], 16) % 2147483647
            request = dict(model=self.model, messages=task['messages'], temperature=.7, top_p=.8,
                           max_tokens=2048, seed=seed, chat_template_kwargs={'enable_thinking': False},
                           response_format={'type': 'json_schema', 'json_schema': {
                               'name': 'trace_diagnostic_explanation', 'strict': True,
                               'schema': gen.Answer.model_json_schema()}})
            tick = time.monotonic()
            raw = self.traced_call('generation',task,'generation',i+1,'chat',request,seed,lambda:self.request(request))
            attempt = dict(attempt=i + 1, seed=seed, raw_response=raw,request_parameters=deepcopy(request),prompt_tokens=count)
            try:
                parsed = gen.parse_response(raw, count)
                attempt['status'] = 'complete'
            except Exception as e:
                attempt.update(status='failed', error=str(e))
            attempt['seconds'] = time.monotonic() - tick; attempts.append(attempt)
            dump(journal, attempts)
            if parsed is not None:
                break
        return dict(task_id=task['task_id'], patient_id=task['patient_id'], status='complete' if parsed else 'failed',
                    parsed_explanation=parsed['explanation'] if parsed else None,
                    parsed_diagnoses=parsed['ranked_diagnoses'] if parsed else None,
                    prompt_tokens=count, attempts=len(attempts))

    def match(self, task):
        attempts = []
        journal = self.out / 'matching_attempts_PRIVATE' / (task['id'] + '.json')
        existing = {(x['mode'], x['attempt']): x for x in read(journal)} if journal.exists() else {}
        def persist():
            # Replaying an earlier saved response must not erase later saved attempts if interrupted.
            merged=dict(existing)
            merged.update({(x['mode'],x['attempt']):x for x in attempts})
            dump(journal,[merged[k] for k in sorted(merged,key=lambda k: (['free','segment'].index(k[0]),k[1]))])
        for mode in ['free', 'segment']:
            module = base if mode == 'free' else segment
            budget = 3072 if mode == 'free' else 2048
            body = dict(facts=[dict(index=i, feature_id=f['feature_id'], state=f['state'], text=f['display_text'])
                               for i, f in enumerate(task['facts'])])
            if mode == 'free':
                body['explanation'] = task['explanation']
            else:
                body['segments'] = [dict(id=s['id'], text=s['text']) for s in segment.segments(task['explanation'])]
            messages = [dict(role='system', content=module.PROMPT),
                        dict(role='user', content=json.dumps(body, ensure_ascii=False))]
            white = False
            for i in range(2):
                saved = existing.get((mode, i + 1))
                n = saved['prompt_tokens'] if saved else self.traced_tokens('matching',task,mode,i+1,messages,budget)
                request = dict(model=self.model, messages=messages,
                               temperature=.2 if white or (mode == 'segment' and i) else 0,
                               max_tokens=budget, chat_template_kwargs={'enable_thinking': False},
                               response_format={'type': 'json_schema', 'json_schema': {
                                   'name': 'trace_match', 'strict': True, 'schema': module.Answer.model_json_schema()}})
                tick = time.monotonic()
                raw = saved['raw_response'] if saved else self.traced_call('matching',task,mode,i+1,'chat',request,None,lambda:self.request(request))
                retry = False
                at = dict(mode=mode, attempt=i + 1, prompt_tokens=n, raw_response=raw,request_parameters=deepcopy(request))
                try:
                    choice = raw['choices'][0]; msg = choice['message']; text = msg.get('content') or ''
                    if choice['finish_reason'] != 'stop':
                        if choice['finish_reason'] == 'length' and base.whitespace_stall(text):
                            white = True; retry = True; raise ValueError('WHITESPACE_STALL')
                        raise ValueError('FINISH_' + str(choice['finish_reason']))
                    if msg.get('reasoning_content') or msg.get('reasoning') or '<think>' in text:
                        raise ValueError('THINKING_OUTPUT')
                    retry = True
                    parsed = module.validate(json.loads(text), task)
                    at['seconds'] = time.monotonic() - tick; attempts.append(at)
                    persist()
                    return dict(id=task['id'], task_id=task['task_id'], status='COMPLETE', parsed=parsed,
                                annotation_protocol='full_v1_' + mode, attempts=len(attempts))
                except Exception as e:
                    at.update(error=str(e), seconds=time.monotonic() - tick); attempts.append(at)
                    persist()
                    if mode == 'free' and not retry:
                        break
                    if white:
                        messages += [dict(role='user', content='Restart as compact JSON; no formatting whitespace outside strings. Include every field.')]
                    else:
                        messages += [dict(role='assistant', content=raw['choices'][0]['message'].get('content') or ''),
                                     dict(role='user', content='Correct this validation error and return the complete small-group schema: ' + str(e)[:700])]
        return dict(id=task['id'], task_id=task['task_id'], status='FAILED', parsed=None,
                    annotation_protocol='full_v1_failed', attempts=len(attempts))

def queue(tasks, key, function, folder, workers, label, progress_hook=None, verify_first=True):
    """One atomic result per task; successful and terminal failed tasks are immutable."""
    folder.mkdir(parents=True, exist_ok=True)
    ids = {t[key] for t in tasks}
    if len(ids) != len(tasks):
        raise ValueError('Duplicate queue IDs')
    cached = {p.stem: read(p) for p in folder.glob('*.json')}
    if not set(cached) <= ids:
        raise ValueError('Foreign queue result')
    task_index = {t[key]: t for t in tasks}
    for identity, r in cached.items():
        if r.get('task_input_digest') != digest(task_index[identity]):
            raise ValueError('Cached task input changed')
    pending = [t for t in tasks if t[key] not in cached]; started = time.monotonic()
    count = len(cached); initial = count; recent = deque(maxlen=50); stopped = None
    def record_exception(t,e):
        event=dict(stage=label,task_key=t[key],task_id=t.get('task_id'),group_id=t.get('id'),
                   time_utc=datetime.now(timezone.utc).isoformat(),error_type=type(e).__name__,
                   infrastructure=isinstance(e,InfrastructureError),http_status=getattr(e,'http_status',None))
        dump(folder.parent/'queue_errors_PRIVATE'/(uuid.uuid4().hex+'.json'),event)
    def save(t, r):
        nonlocal count
        if r[key] != t[key]:
            raise ValueError('Response/task mismatch')
        r = dict(r, task_input_digest=digest(t))
        dump(folder / (t[key] + '.json'), r); cached[t[key]] = r; count += 1
        recent.append(r['status'] not in ['complete', 'COMPLETE'])
    def progress(inflight):
        elapsed = time.monotonic() - started; rate = (count - initial) / max(elapsed, .001) * 3600
        result = dict(stage=label, completed=count, planned=len(tasks), cached=initial, in_flight=inflight,
                      statuses=dict(Counter(r['status'] for r in cached.values())), elapsed_seconds=round(elapsed, 1),
                      tasks_per_hour=round(rate, 1), eta_hours=(len(tasks) - count) / rate if rate else None)
        if progress_hook is not None:
            result.update(progress_hook(len(tasks)-count, elapsed, rate))
        dump(folder.parent / (label + '_progress.json'), result)
        print(json.dumps(result), flush=True)
    progress(0)
    if pending and verify_first:
        # One actual request verifies schema support before 128-way dispatch.
        first = pending.pop(0)
        try: save(first, function(first))
        except Exception as e:
            record_exception(first,e);progress(0);raise
        progress(0)
        if recent[-1]:
            raise RuntimeError(label + ' first request failed validation; inspect PRIVATE attempts')
    if pending:
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            active = {}; iterator = iter(pending); exhausted = False; last = time.monotonic()
            while active or not exhausted:
                while not stopped and not exhausted and len(active) < workers:
                    t = next(iterator, None)
                    if t is None:
                        exhausted = True; break
                    active[pool.submit(function, t)] = t
                if not active:
                    break
                finished, _ = cf.wait(active, timeout=20, return_when=cf.FIRST_COMPLETED)
                for future in finished:
                    t = active.pop(future)
                    try:
                        save(t, future.result())
                    except Exception as e:
                        record_exception(t,e)
                        stopped = e
                    if (len(recent) >= 5 and all(list(recent)[-5:])) or (len(recent) == 50 and sum(recent) >= 10):
                        stopped = stopped or RuntimeError(label + ' failure circuit opened; inspect PRIVATE attempts')
                if time.monotonic() - last >= 20 or not active:
                    progress(len(active)); last = time.monotonic()
            if stopped:
                raise stopped
    if set(cached) != ids:
        raise RuntimeError('Incomplete queue ' + label)
    progress(0)
    return [cached[t[key]] for t in tasks]
