"""One shared task queue, 128 concurrent tasks per identical local model service."""
import concurrent.futures as cf
import math
import re
import threading
import time
from pathlib import Path
from .common import read, dump, digest
from .service import Client, service, InfrastructureError


def endpoints(path):
    config = read(path)
    nodes = config['nodes']
    if not nodes or not 1 <= len(nodes) <= 32:
        raise ValueError('Configure 1..32 local model nodes')
    if len({n['name'] for n in nodes}) != len(nodes) or len({n['endpoint'].rstrip('/') for n in nodes}) != len(nodes):
        raise ValueError('Duplicate node name or endpoint; do not list one service several times')
    for n in nodes:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', n['name']):
            raise ValueError('Node names must contain letters, numbers, underscore or hyphen')
        if 'NODE_' in n['endpoint'] or '<' in n['endpoint']:
            raise ValueError('Fill the actual node addresses in --endpoints-file before running')
    return nodes


class Pool:
    """Every dispatched task owns its journal throughout failover; no duplicate active task."""
    def __init__(self, nodes, clients, capacity=128, out=None, budget_start=None,
                 target_minutes=120, local_reserve_minutes=30):
        self.nodes = nodes; self.clients = clients; self.capacity = capacity; self.out = out
        self.condition = threading.Condition()
        self.live = [True] * len(nodes); self.active = [0] * len(nodes)
        self.completed = [0] * len(nodes); self.failed = [0] * len(nodes)
        self.routed = [0] * len(nodes); self.network_failures = [0] * len(nodes)
        self.stage = 'unstarted'; self.stage_started = time.monotonic()
        self.budget_start = budget_start if budget_start is not None else time.monotonic()
        self.target_minutes = target_minutes; self.reserve_minutes = local_reserve_minutes
        self.events = []

    def begin(self, stage):
        with self.condition:
            if any(self.active):
                raise RuntimeError('Cannot change API stage with active requests')
            self.stage = stage; self.stage_started = time.monotonic()
            self.completed = [0] * len(self.nodes); self.failed = [0] * len(self.nodes)
            self.routed = [0] * len(self.nodes)

    def acquire(self, preferred=None):
        with self.condition:
            while True:
                candidates = [i for i in range(len(self.nodes)) if self.live[i] and self.active[i] < self.capacity]
                if preferred is not None:
                    candidates = [i for i in candidates if i == preferred]
                    if not self.live[preferred]:
                        raise InfrastructureError('Requested warmup node unavailable')
                if candidates:
                    # Equal in-flight work at dispatch; faster services naturally receive more tasks.
                    i = min(candidates, key=lambda j: (self.active[j], self.routed[j], j))
                    self.active[i] += 1; self.routed[i] += 1
                    return i
                if not any(self.live):
                    raise InfrastructureError('All inference nodes unavailable; resume after restoring service')
                self.condition.wait(timeout=5)

    def execute(self, task, preferred=None):
        history = []
        for _ in range(len(self.nodes)):
            i = self.acquire(preferred)
            try:
                result = getattr(self.clients[i], self.stage)(task)
                with self.condition:
                    self.completed[i] += 1
                    if result['status'] not in ['complete','COMPLETE']:
                        self.failed[i] += 1
                return dict(result, worker_node=self.nodes[i]['name'], transport_failover_nodes=history)
            except InfrastructureError as e:
                history.append(self.nodes[i]['name'])
                with self.condition:
                    self.network_failures[i] += 1; self.live[i] = False
                    event = dict(node=self.nodes[i]['name'],stage=self.stage,error=str(e),action='node_paused_pending_tasks_reassigned')
                    self.events.append(event)
                    if self.out is not None:
                        dump(self.out/'node_failover.json',self.events)
                print('NODE_PAUSED',self.nodes[i]['name'],str(e),flush=True)
                if preferred is not None:
                    raise
            finally:
                with self.condition:
                    self.active[i] -= 1; self.condition.notify_all()
        raise InfrastructureError('No healthy node remains for task')

    def warmup(self, tasks, key, folder):
        """One real pending task per node, saved to the same cache as the main queue."""
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
        pending = [t for t in tasks if t[key] not in cached]
        enabled = [i for i in range(len(self.nodes)) if self.live[i]]
        jobs = list(zip(enabled, pending))
        if not jobs:
            return
        print('API_WARMUP', self.stage, 'nodes=', len(jobs), 'uses pending production tasks', flush=True)
        errors = []
        with cf.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            active = {executor.submit(self.execute, t, preferred=i): (i, t) for i, t in jobs}
            while active:
                finished, _ = cf.wait(active, timeout=20, return_when=cf.FIRST_COMPLETED)
                if not finished:
                    print('API_WARMUP waiting', self.stage, [self.nodes[i]['name'] for i, _ in active.values()], flush=True)
                for future in finished:
                    i, t = active.pop(future)
                    try:
                        r = future.result()
                        if r[key] != t[key]:
                            raise ValueError('Response/task mismatch')
                        dump(folder / (t[key] + '.json'), dict(r, task_input_digest=digest(t)))
                        if r['status'] not in ['complete', 'COMPLETE']:
                            raise RuntimeError('Warmup task failed validation; inspect PRIVATE attempts')
                        print('API_WARMUP passed', self.stage, self.nodes[i]['name'], flush=True)
                    except Exception as error:
                        errors.append(error)
        if errors:
            raise errors[0]

    def progress(self, remaining, elapsed, rate):
        with self.condition:
            t = max(time.monotonic()-self.stage_started,.001)
            nodes = [dict(node=n['name'],enabled=self.live[i],in_flight=self.active[i],
                          completed=self.completed[i],terminal_failed=self.failed[i],
                          infrastructure_failures=self.network_failures[i],
                          completed_per_hour=round(self.completed[i]/t*3600,1)) for i,n in enumerate(self.nodes)]
            active = sum(self.live)
        budget_elapsed = time.monotonic()-self.budget_start
        available = self.target_minutes*60-budget_elapsed-self.reserve_minutes*60
        eta = remaining/rate*3600 if rate>0 else None
        estimate = dict(stage=self.stage,target_total_minutes=self.target_minutes,
                        runtime_elapsed_minutes=round(budget_elapsed/60,1),local_work_reserve_minutes=self.reserve_minutes,
                        remaining_stage_minutes=round(eta/60,1) if eta is not None else None,
                        forecast_includes='Current API stage and reserved local work; generation-stage forecast excludes matching')
        if self.stage == 'match' and elapsed>=600 and eta is not None:
            projected=(budget_elapsed+eta+self.reserve_minutes*60)/60
            estimate.update(projected_total_minutes=round(projected,1),target_on_track=projected<=self.target_minutes)
            if available>0 and rate>0:
                estimate['estimated_nodes_needed_at_observed_rate']=max(active,math.ceil(active*eta/available))
            else:
                estimate['estimated_nodes_needed_at_observed_rate']=None
            estimate['action']='continue' if projected<=self.target_minutes else 'increase_identical_nodes_and_resume; no results are discarded'
        return dict(nodes=nodes,node_capacity=self.capacity,active_nodes=active,deadline_estimate=estimate)


def setup(path, model, model_dir, out, budget_start, target_minutes=120, reserve_minutes=30):
    nodes=endpoints(path)
    print('Checking',len(nodes),'services; same checkpoint required; 128 tasks per node',flush=True)
    def check(n):
        folder=out/'node_services'/n['name'];folder.mkdir(parents=True,exist_ok=True)
        limit=service(n['endpoint'],model,model_dir,folder)
        return n['name'],limit
    with cf.ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        limits=dict(pool.map(check,nodes))
    if len(set(limits.values()))!=1:
        raise ValueError('All services must advertise the same context limit')
    snapshot=dict(nodes=nodes,model=model,workers_per_node=128,total_api_workers=len(nodes)*128,
                  context_limits=limits,weight_identity='Operator deploys the same checkpoint on all nodes; coordinator metadata is captured')
    # Topology can grow between invocations; scientific tasks and cached results remain fixed.
    dump(out/'node_topology.json',snapshot)
    runtime=out/'runtime';runtime.mkdir(exist_ok=True)
    dump(runtime/('nodes_'+str(time.time_ns())+'.json'),snapshot)
    clients=[Client(n['endpoint'],model,limits[n['name']],out) for n in nodes]
    return Pool(nodes,clients,128,out,budget_start,target_minutes,reserve_minutes)
