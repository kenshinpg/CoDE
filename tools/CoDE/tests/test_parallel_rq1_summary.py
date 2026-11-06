import sys
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from code_eval.parallel_rq1_summary import interval_batch, summarize_pair
from code_eval.trace2400 import rq1_summary as trace
from code_eval import rq1_summary as ddx


class ParallelSummaryContracts(unittest.TestCase):
    def test_cluster_draws_and_patient_means_match_exactly(self):
        values=[.3,-.1,1.,0.,.2];groups=["a","a","b","c","d"]
        got=interval_batch((tuple(groups),261013,53,[(0,values)]))[0][1]
        self.assertEqual(got,trace.interval(values,bootstrap=53,groups=groups))
        got=interval_batch((tuple(range(len(values))),261006,53,[(0,values)]))[0][1]
        self.assertEqual(got,ddx.interval(values,bootstrap=53))

    def test_parallel_before_after_matches_serial_full_schema(self):
        rows=[]
        methods=sorted(set(m for ms in trace.FAMILIES.values() for m in ms))
        for i in range(4):
            for p in ["encoder","lightgbm"]:
                for budget in trace.BUDGETS:
                    for j,m in enumerate(methods):
                        rows.append(dict(patient_id=str(i),subject_group="a" if i<2 else str(i),
                          probe=p,fraction=budget,selector=m,available=not(i==3 and m=="tabr"),
                          both_full_correct=i<3,correct=int((i+j)%3!=0),random_accuracy=(i+j)%7/7,
                          fact_count=2,selected_tokens=[str(i),str(j)]))
        new=[dict(r,random_accuracy=r["random_accuracy"]*.9) if r["fraction"]==.25 else dict(r) for r in rows]
        expected=(trace.summarize(rows,bootstrap=17),trace.summarize(new,bootstrap=17))
        actual=summarize_pair(rows,new,"trace",workers=2,bootstrap=17)
        self.assertEqual(actual,expected)

if __name__=="__main__":unittest.main()
