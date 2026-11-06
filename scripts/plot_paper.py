"""Replot accepted aggregate statistics. No inference, model or API call."""
from pathlib import Path
import os,json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'build/figures';B.mkdir(parents=True,exist_ok=True)
PANELS=json.loads((ROOT/'results/figure_inputs/accepted_panels.json').read_text())
controlled=PANELS['controlled'];cont=PANELS['continuous']
keys=[('cohd','Association'),('encoder','Representation'),('grasp','Group-enhanced'),('lightgbm','Tree removal'),('tabr','TabR-S'),('tree_gain','Global gain'),('tree_local_abs_shap','Local absolute SHAP'),('tree_class_abs_shap','Class SHAP'),('no_retrieval','No-retrieval')]
def one(rows,**kw):
    found=[r for r in rows if all(r.get(k)==v for k,v in kw.items())]
    if len(found)!=1:raise ValueError((kw,len(found)))
    return found[0]
def sel(ds,probe,key,label,fraction):
    family='tree_rules' if key.startswith('tree_') else 'retrieval' if key=='no_retrieval' else 'sources'
    r=one(PANELS['selection'][ds],probe=probe,selector=key,family=family,population='both_full_correct',fraction=fraction)['gain']
    return tuple(100*v for v in [r['mean'],*r['ci95']])
def ndiff(ds,probe,m,b):
    p=one(PANELS['natural'][ds],probe=probe,family=f'paired_{m}_vs_{b}',stratum='correct',endpoint='reference')['paired_ap_differences'][m+' minus '+b]
    return p['difference'],*p['ci95']
def binary(ds,fraction):
    p=one(controlled[ds],probe='encoder',family='paired_tree_positive_shap_vs_lightgbm_absolute',stratum='all',endpoint='reference',fraction=fraction)
    v=p['information_audit']['paired_concordance_differences']['tree_positive_shap minus lightgbm_absolute']
    return 100*v['difference'],*[100*x for x in v['ci95']],p['pair_patients']
def save(fig,name):
    fig.canvas.draw()
    fig.savefig(B/(name+'.pdf'));fig.savefig(B/(name+'.svg'));fig.savefig(B/(name+'.png'),dpi=160)
    plt.close(fig)
def cap(*args,**kwargs):pass
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':True})
# One comparison grid per dataset and probe, with identical scales and geometry.
fig,axs=plt.subplots(2,2,figsize=(10.5,8.0))
fig.subplots_adjust(left=.22,right=.98,bottom=.08,top=.94,wspace=.90,hspace=.42)
for ri,dataset in enumerate(['DDXPlus','TRACE']):
 for ci,probe in enumerate(['encoder','lightgbm']):
    ax=axs[ri,ci];data=np.array([[sel(dataset,probe,k,n,f)[0] for f in [.25,.5,.75]] for k,n in keys])
    ax.imshow(data,cmap='RdBu',norm=TwoSlopeNorm(vmin=-35,vcenter=0,vmax=35),aspect='auto')
    for i,(k,n) in enumerate(keys):
     for j,f in enumerate([.25,.5,.75]):
        v,lo,hi=sel(dataset,probe,k,n,f)
        star=probe==k or (probe=='lightgbm' and (k.startswith('tree_') or k=='lightgbm'))
        ax.text(j,i,f'{v:+.2f}'+('†' if lo<=0<=hi else '')+('*' if star else ''),ha='center',va='center',fontsize=8,color='white' if abs(v)>23 else '#16222b')
    ax.set_xticks(range(3),['25%','50%','75%']);ax.set_yticks(range(len(keys)),[n for k,n in keys],fontsize=8)
    ax.tick_params(length=0);ax.set_xlabel('Group budget')
    ax.set_title(f'{chr(97+ri*2+ci)}  {dataset} · '+('Representation' if ci==0 else 'Tree')+f' (n={1490 if ri==0 else 251:,})',loc='left',fontsize=9,weight='bold')
save(fig,'fig02_rq1')
cap(2,'fig02_rq1','fig:evidence-curves',r'Evidence-selection gains in DDXPlus and TRACE. Values are percentage-point gains over each selector\textquotesingle s fact-count-matched random controls, on a shared zero-centered scale. $\dagger$ marks a paired 95\% interval containing zero; * marks self-evaluation. Group-enhanced shares the Representation architecture. The primary populations contain 1,490 DDXPlus patients and 251 TRACE admissions. Exact intervals and family-specific all-test sensitivities are in Supplementary Sections~\ref*{sec:rq1-results} and~\ref*{sec:trace-results}. Differences between cells are not paired method contrasts.')


# Preserve the submitted three-panel grammar; add a matched TRACE row.
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,'axes.titlesize':12,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':10})
blue='#245b8e';light='#9fc4de';gray='#738494'
fig,axs=plt.subplots(2,3,figsize=(14.3,8.5))
fig.subplots_adjust(left=.17,right=.985,bottom=.085,top=.89,wspace=.65,hspace=.67)
selection=[('tree_local_abs_shap','Local absolute SHAP','Local |SHAP|'),('tree_class_abs_shap','Class SHAP','Class SHAP'),('grasp','Group-enhanced','Group-enhanced removal'),('cohd','Association','Association removal'),('tabr','TabR-S','Similar-case removal'),('lightgbm','Tree removal','Tree removal'),('tree_gain','Global gain','Global gain')]
source=[]
for row,ds in enumerate(['DDXPlus','TRACE']):
 a=axs[row,0]
 for i,(key,label,display) in enumerate(selection):
  v,lo,hi=sel(ds,'encoder',key,label,.25)
  a.barh(i,v,color=blue if lo>0 else gray,height=.59)
  a.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='none',ecolor='#182d42',capsize=3,lw=1)
  a.text(hi+.45,i,f'{v:+.2f}'.replace('-', '−'),va='center',fontsize=10.5)
  source.append(dict(panel=f'{row}a',method=key,mean=v,ci=[lo,hi]))
 a.set_yticks(range(7),[x[2] for x in selection]);a.set_ylim(6.65,-.65);a.set_xlim(-3,48);a.set_xticks([0,10,20,30,40])
 a.axvline(0,color='#7d8790',lw=.8);a.grid(False);a.set_axisbelow(True)
 a.set_xlabel('Accuracy gain over matched random (pp)')
 a.set_title(f'({chr(97+3*row)}) Selecting diagnostic evidence',loc='left',weight='bold',pad=25)
 a.text(0,1.025,'25% group budget · '+('1490 patients' if row==0 else '251 admissions'),transform=a.transAxes,fontsize=10.5,color='#59616b')
 a=axs[row,1]
 for j,(probe,col,marker,name) in enumerate([('encoder',blue,'o','Representation probe'),('lightgbm',gray,'s','Tree probe')]):
  for i,m in enumerate(['cohd_coverage','tree_positive_shap']):
   v,lo,hi=ndiff(ds,probe,m,'group_coverage')
   a.errorbar(v,i+(j-.5)*.24,xerr=[[v-lo],[hi-v]],fmt=marker,color=col,capsize=3,ms=5,label=name if i==0 else None)
   source.append(dict(panel=f'{row}b',probe=probe,method=m,mean=v,ci=[lo,hi]))
 a.set_yticks([0,1],['Association\ncoverage','Local +SHAP\ncoverage*']);a.set_ylim(1.75,-.65);a.set_xlim(-.16,.51);a.set_xticks([-.1,0,.1,.2,.3,.4,.5])
 a.axvline(0,color='#7d8790',ls='--',lw=.8);a.grid(False)
 a.set_xlabel('Failure-AP difference vs. group coverage')
 a.set_title(f'({chr(98+3*row)}) Evaluating natural rationales',loc='left',weight='bold',pad=25)
 a.text(0,1.025,'1979 rationales · 20 R / 39 T failures' if row==0 else '614 rationales · 105 R / 44 T failures',transform=a.transAxes,fontsize=10.5,color='#59616b')
 a.legend(loc='lower right',frameon=False,fontsize=10.5)
 a=axs[row,2];ticks=[]
 for i,f in enumerate([.25,.5,.75]):
  p=one(controlled[ds],probe='encoder',family='paired_tree_positive_shap_vs_lightgbm_absolute',stratum='all',endpoint='reference',fraction=f)
  for j,(m,col,label) in enumerate([('lightgbm_absolute',blue,'Direct support A'),('tree_positive_shap',light,'Weighted coverage')]):
   v=100*p['methods'][m]['pair_concordance'];x=i+(j-.5)*.37
   a.bar(x,v,width=.30,color=col,label=label if i==0 else None)
   a.text(x+(.055 if j else -.055),100*max(p['methods'][mm]['pair_concordance'] for mm in ['lightgbm_absolute','tree_positive_shap'])+2,f'{v:.1f}',ha='center',fontsize=10.5)
   source.append(dict(panel=f'{row}c',budget=f,method=m,concordance=v))
  v,lo,hi,n=binary(ds,f);v,lo,hi=-v,-hi,-lo
  a.text(i,112,f'Δ {v:+.2f}\n[{lo:.2f},{hi:.2f}]'.replace('-', '−'),ha='center',va='center',fontsize=9.4)
  ticks.append(f'{int(f*100)}%\nn = {n}')
 a.set_ylim(-40,124);a.set_xlim(-.65,2.65);a.set_yticks([0,25,50,75,100]);a.set_xticks(range(3),ticks)
 a.set_ylabel('Patient-averaged concordance (%)');a.set_xlabel('Controlled evidence budget')
 a.set_title(f'({chr(99+3*row)}) Scoring evidence jointly',loc='left',weight='bold',pad=25)
 a.text(0,1.025,'Equal facts · Representation probe',transform=a.transAxes,fontsize=10.5,color='#59616b')
 a.grid(False);a.set_axisbelow(True)
 a.legend(loc='lower left',frameon=True,facecolor='white',edgecolor='none',fontsize=10.5)
fig.text(.035,.925,'DDXPlus',fontsize=15,weight='bold',color=blue)
fig.text(.035,.438,'TRACE',fontsize=15,weight='bold',color=blue)
save(fig,'fig01_overview')
(B/'fig01_source_data.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
cap(1,'fig01_overview','fig:code-overview',r'From evidence selection to rationale evaluation in DDXPlus (top) and TRACE (bottom). (a,d) Selection gains at the 25\% group budget under the Representation probe, relative to each selector\textquotesingle s fact-count-matched random control. SHAP and global gain use the Tree source; Group-enhanced shares the Representation backbone. Similar-case denotes TabR-S. (b,e) Correct-proposal failure-AP differences from uniform group coverage under Representation (R) and Tree (T) probes. Scores evaluate automatically mapped findings cited by the LLM. * marks Tree-probe local-SHAP self-evaluation. Intervals in (a,b,d,e) are paired 95\% bootstrap intervals; AP contrasts are interpreted within datasets. (c,f) Tree direct support and local positive-SHAP coverage on equal-fact-count controlled sets under the Representation probe. Annotations give direct support minus coverage (percentage points) with paired 95\% intervals; $n$ counts contributing patients/admissions with discordant pairs. The 75\% differences remain uncertain in both datasets. Corresponding columns use common axis ranges. Incorrect- and all-proposal sensitivities are reported separately.')
print('Overview reconstructed from accepted panels.')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
fig,axs=plt.subplots(2,2,figsize=(10.5,5.9))
fig.subplots_adjust(left=.17,right=.98,bottom=.12,top=.93,wspace=.65,hspace=.73)
for ri,dataset in enumerate(['DDXPlus','TRACE']):
 for ci,probe in enumerate(['encoder','lightgbm']):
    ax=axs[ri,ci];matrix=np.zeros((2,3))
    for i,m in enumerate(['cohd_coverage','tree_positive_shap']):
     for j,b in enumerate(['group_coverage','cohd_absolute','lightgbm_absolute']):matrix[i,j]=ndiff(dataset,probe,m,b)[0]
    ax.imshow(matrix,cmap='RdBu',norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),aspect='auto')
    for i,m in enumerate(['cohd_coverage','tree_positive_shap']):
     for j,b in enumerate(['group_coverage','cohd_absolute','lightgbm_absolute']):
        v,lo,hi=ndiff(dataset,probe,m,b);star=probe=='lightgbm' and (i==1 or j==2)
        ax.text(j,i,f'{v:+.3f}'.replace('-', '−')+('*' if star else '')+f'\n[{lo:+.3f}, {hi:+.3f}]'.replace('-', '−'),ha='center',va='center',fontsize=7.5,color='white' if abs(v)>.6 else '#16222b')
    ax.set_xticks(range(3),['Group\ncoverage','Association\nA','Tree\nA']);ax.set_yticks(range(2),['Association\ncoverage','Local positive\nSHAP']);ax.tick_params(length=0)
    fail=[[20,39],[105,44]][ri][ci]
    ax.set_title(f'{chr(97+ri*2+ci)}  {dataset} · '+('Representation' if ci==0 else 'Tree')+f'\n{fail} failures / {1979 if ri==0 else 614} explanations',loc='left',fontsize=9,weight='bold')
save(fig,'fig03_rq2')
cap(3,'fig03_rq2','fig:natural-effects',r'Natural-rationale failure-AP contrasts depend on comparator and probe. Cells give coverage minus comparator AP with paired 95\% cluster-bootstrap intervals, on the same $[-1,1]$ scale. Each comparison uses common eligible correct proposals: 1,979 explanations/889 patients in DDXPlus and 614/210 admissions in TRACE. * marks comparisons involving Tree-derived scores under the Tree probe. Inputs are record-compatible cited findings plus the dataset-specific fixed context. Automatic mappings supply the main analysis; reviewer-policy sensitivity is reported separately. Failure prevalence differs between datasets.')


fig,axs=plt.subplots(2,2,figsize=(10.5,6.7))
fig.subplots_adjust(left=.11,right=.98,bottom=.10,top=.94,wspace=.25,hspace=.65)
blue='#3b6d93';orange='#b27b47'
for ci,dataset in enumerate(['DDXPlus','TRACE']):
    ax=axs[0,ci]
    for i,f in enumerate([.25,.5,.75]):
        v,lo,hi,n=binary(dataset,f);ax.errorbar(i,v,yerr=[[v-lo],[hi-v]],fmt='o',color=blue,capsize=4)
        ax.text(i,-15.5,f'n={n}',ha='center',fontsize=9)
    ax.set_ylim(-18,12);ax.set_title(f'{chr(97+ci)}  {dataset}: binary recovery',loc='left',weight='bold',fontsize=10)
for ci,subset in enumerate(['all','both_correct']):
    ax=axs[1,ci]
    for i,f in enumerate([.25,.5,.75]):
        p=one(cont,kind='controlled',probe='encoder',endpoint='q',gap=.01,fraction=f,subset=subset)
        for j,(m,col,label) in enumerate([('lightgbm_absolute',blue,'Tree A'),('cohd_absolute',orange,'Association A')]):
            v=p['paired_differences']['tree_positive_shap minus '+m];mu=100*v['mean'];lo,hi=[100*x for x in v['ci95']]
            ax.errorbar(i+(j-.5)*.18,mu,yerr=[[mu-lo],[hi-mu]],fmt='o',color=col,capsize=3,label=label if i==0 else None)
        ax.text(i,-15.5,f"n={p['patients']}",ha='center',fontsize=9)
    ax.set_ylim(-18,12);ax.set_title(f'{chr(99+ci)}  DDXPlus: '+('continuous support' if ci==0 else 'both subsets correct'),loc='left',weight='bold',fontsize=10)
    if ci==0:fig.legend(*ax.get_legend_handles_labels(),fontsize=8,loc='lower center',bbox_to_anchor=(.5,0),ncol=2,frameon=False)
for ax in axs.flat:
    ax.axhline(0,color='#7d858c',lw=.8);ax.set_yticks([-10,0,10]);ax.set_xticks(range(3),['25%','50%','75%']);ax.set_xlim(-.35,2.35);ax.set_xlabel('Group budget');ax.grid(axis='y',alpha=.12)
for ax in axs[:,0]:ax.set_ylabel('SHAP − direct support\nConcordance difference (pp)')
save(fig,'fig04_rq3')
cap(4,'fig04_rq3','fig:controlled-effects',r'Controlled equal-fact-count comparisons under the Representation probe. Positive differences favor local positive-SHAP coverage. (a,b) Binary success--failure concordance relative to Tree direct support in DDXPlus and TRACE. (c,d) DDXPlus exploratory continuous-support comparisons with target-probability differences $\geq0.01$, for all eligible pairs and pairs where both subsets recover the reference diagnosis. Points and bars give mean paired differences and 95\% bootstrap intervals; $n$ denotes contributing patients/admissions. Continuous and binary analyses have different eligible pairs. No TRACE continuous-support result is implied.')



# Exploratory DDXPlus intervention ECDFs. Source arrays preserve patient means.
effects=json.loads((ROOT/'results/figure_inputs/ddx_intervention_effects.json').read_text())
fig,axs=plt.subplots(2,2,figsize=(10.5,6.5),sharex=True,sharey=True)
colors={'cohd':'#245b8e','local_shap':'#bc7944','class_shap':'#518476'}
for ax,(op,stratum) in zip(axs.flat,[('add','correct'),('delete','correct'),('add','wrong'),('delete','wrong')]):
    for key,values in effects.items():
        selector,direction,operation,proposal,probe=key.split('|')
        if probe!='encoder' or direction not in ['support','unsigned'] or operation!=op or proposal!=stratum:continue
        arr=np.sort(values);ax.step(arr,np.arange(1,len(arr)+1)/len(arr),where='post',label=selector,color=colors.get(selector),lw=1.2)
    ax.set_xscale('symlog',linthresh=.1);ax.axvline(0,color='#8b9399',lw=.7)
    ax.set_title(f'{stratum.capitalize()} proposals: {op}');ax.set_xlabel('Effect over matched random (probability pp)');ax.set_ylabel('Patient fraction');ax.grid(alpha=.15)
axs[0,0].legend(fontsize=7,frameon=False)
fig.tight_layout();save(fig,'supplement_intervention_ecdf')
print('Five numeric figures written to',B)
