"""Infrastructure recovery counts are separate from returned-response/semantic failures."""
from collections import Counter
from .common import read,dump,emit


def execution_summary(folder, write=True):
    events=[read(p) for p in sorted((folder/'infrastructure_events_PRIVATE').glob('*.json'))]
    result={}; unresolved=[]; decorated=[]
    for stage in ['generation','matching']:
        cache={p.stem:read(p) for p in (folder/(stage+'_results_PRIVATE')).glob('*.json')}
        journals=[read(p) for p in (folder/(stage+'_attempts_PRIVATE')).glob('*.json')]
        ev=[e for e in events if e['stage']==stage]
        ids={e['group_id'] or e['task_id'] for e in ev}
        pending=sorted(ids-set(cache))
        unresolved += [dict(stage=stage,task_key=x) for x in pending]
        result[stage]=dict(terminal_results=len(cache),terminal_statuses=dict(Counter(r['status'] for r in cache.values())),
            returned_response_attempts=sum(len(j) for j in journals),infrastructure_events=len(ev),
            infrastructure_affected_tasks=len(ids),infrastructure_unfinished_tasks=len(pending))
        for e in ev:
            k=e['group_id'] or e['task_id']
            decorated.append(dict(e,terminal_result_available=k in cache,
                                  terminal_status=cache[k]['status'] if k in cache else None))
    result['unresolved_infrastructure_tasks']=unresolved
    result['policy']='Infrastructure events do not consume the fixed returned-response format budget; resume independent of outcomes'
    if write:
        dump(folder/'execution_summary.json',result)
        emit(folder/'infrastructure_events_summary_PRIVATE.jsonl',decorated)
    return result


def aggregate_execution(out):
    totals=Counter();nodes=[]
    for i in range(8):
        s=execution_summary(out/'nodes'/f'node_{i}',write=False);nodes.append(dict(node=i,summary=s))
        for stage in ['generation','matching']:
            for key in ['returned_response_attempts','infrastructure_events','infrastructure_affected_tasks','infrastructure_unfinished_tasks']:
                totals[stage+'_'+key]+=s[stage][key]
    result=dict(totals=dict(totals),nodes=nodes,
        note='Affected-task counts overlap across stages. Infrastructure-unfinished tasks are not semantic failures.')
    dump(out/'infrastructure_summary.json',result)
    return result


def measurement_flow(ledger, matches, infrastructure):
    return dict(planned_explanations=len(ledger),
        generation_terminal_failed=sum(e['generation_status']=='failed' for e in ledger),
        matching_terminal_failed_explanations=sum(e['failed_groups']>0 for e in matches),
        matching_missing_group_explanations=sum(e['missing_groups']>0 for e in matches),
        complete_scan_ambiguous_explanations=sum(e['unavailability_reason']=='ambiguous_decisions' for e in ledger),
        complete_mapping_explanations=sum(e['analysis_mapping_available'] for e in ledger),
        raw_scan_statuses=dict(Counter(e['visible_fact_scan_status'] for e in ledger)),
        infrastructure_generation_unfinished_tasks=infrastructure['totals'].get('generation_infrastructure_unfinished_tasks',0),
        infrastructure_matching_unfinished_groups=infrastructure['totals'].get('matching_infrastructure_unfinished_tasks',0),
        formal_population='Further proposal/probe/common-score filters are reported in population_flow.csv; failure categories can overlap')
