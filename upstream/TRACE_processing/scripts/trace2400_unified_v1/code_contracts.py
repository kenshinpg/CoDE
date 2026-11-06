"""Pure interface checks. No semantic repair, fitting or external calls."""
from collections import Counter

ASSERTIONS={x:x.lower() for x in ('PRESENT','ABSENT','FAMILY','HYPOTHETICAL','POSSIBLE')}

def validate(documents, observations):
    errors=[];docs={};obs={};groups={}
    def check(ok,code):
        if not ok:errors.append(code)
    for i,d in enumerate(documents):
        k=d.get('document_id');tag=f'document[{i}]'
        check(isinstance(k,str) and bool(k) and k not in docs,tag+': invalid_or_duplicate_id')
        check(bool(d.get('admission_id')),tag+': missing_admission')
        docs[k]=d
    for i,o in enumerate(observations):
        tag=f'observation[{i}]';oid=o.get('observation_id');aid=o.get('admission_id')
        check(isinstance(oid,str) and bool(oid) and oid not in obs,tag+': invalid_or_duplicate_id');obs[oid]=o
        check(bool(aid),tag+': missing_admission')
        source=o.get('source_type');check(source in ('history','examination','radiology','laboratory'),tag+': unsupported_source')
        did=o.get('document_id');d=docs.get(did)
        if source!='laboratory' or did is not None:
            check(d is not None,tag+': missing_document')
            if d:
                check(d.get('admission_id')==aid,tag+': cross_admission_document')
                check(d.get('source_type')==source,tag+': source_mismatch')
        if source in ('history','examination','radiology') and d:
            r=o.get('raw',{});a=r.get('start');b=r.get('end');t=d.get('raw_text')
            valid=type(a) is int and type(b) is int and isinstance(t,str) and 0<=a<b<=len(t)
            check(valid,tag+': invalid_span')
            if valid:check(t[a:b]==r.get('quote'),tag+': quote_mismatch')
        assertion=o.get('attributes',{}).get('assertion',{})
        check(all(k in assertion for k in ('raw_value','value','status','basis','mapping_version')),tag+': incomplete_assertion')
        raw=assertion.get('raw_value');v=assertion.get('value');status=assertion.get('status')
        if raw in ASSERTIONS:check(v==ASSERTIONS[raw] and status=='normalized',tag+': assertion_mapping')
        elif raw is None:check(v is None and status in ('missing','not_applicable'),tag+': null_assertion')
        else:check(v is None and status=='unrecognized',tag+': unknown_assertion')
        time=o.get('time',{});basis=time.get('basis')
        check(all(k in time for k in ('raw','parsed','basis','source_document_id','missing_reason')),tag+': incomplete_time')
        check(basis in ('missing','inherited_document','native_observation'),tag+': time_basis')
        if basis=='missing':check(time.get('raw') is None and time.get('parsed') is None and bool(time.get('missing_reason')),tag+': missing_time')
        if basis=='inherited_document':
            parent=docs.get(time.get('source_document_id'))
            check(parent is not None and time.get('source_document_id')==did,tag+': inherited_document')
            if parent:
                pt=parent.get('time',{})
                check(pt.get('raw') is not None or pt.get('parsed') is not None,tag+': inherited_empty_time')
                check(all(time.get(k)==pt.get(k) for k in ('raw','parsed')),tag+': inherited_time_mismatch')
        if basis=='native_observation':check(time.get('raw') is not None,tag+': missing_native_time')
        gid=o.get('representation_group_id');check(bool(gid),tag+': missing_representation_group')
        if gid in groups:check(groups[gid]==aid,tag+': cross_admission_group')
        groups[gid]=aid
    graph={}
    for i,o in enumerate(observations):
        deps=o.get('derived_from_observation_ids');check(isinstance(deps,list),f'observation[{i}]: missing_dependencies')
        deps=deps if isinstance(deps,list) else [];graph[o.get('observation_id')]=deps
        for dep in deps:
            parent=obs.get(dep)
            check(parent is not None and dep!=o.get('observation_id'),f'observation[{i}]: invalid_dependency')
            if parent:check(parent.get('admission_id')==o.get('admission_id') and parent.get('representation_group_id')==o.get('representation_group_id'),f'observation[{i}]: dependency_group_mismatch')
    # Iterative cycle check avoids recursion depth dependence on production size.
    indegree={k:0 for k in graph};children={k:[] for k in graph}
    for child,deps in graph.items():
        for parent in deps:
            if parent in graph:indegree[child]+=1;children[parent].append(child)
    queue=[k for k,v in indegree.items() if v==0];count=0
    while queue:
        k=queue.pop();count+=1
        for child in children[k]:
            indegree[child]-=1
            if indegree[child]==0:queue.append(child)
    check(count==len(graph),'dependency_cycle')
    used={o.get('document_id') for o in observations}
    return dict(passed=not errors,errors=errors,documents=len(documents),observations=len(observations),
        source_counts=dict(Counter(o.get('source_type') for o in observations)),
        documents_without_observations=sum(d.get('document_id') not in used for d in documents),
        scope='Selected linkage, span, assertion, time and representation-dependency checks; not full schema or semantic acceptance.')
