"""Fixed original tolerance for TabR query-layout invariance."""
import math
TOLERANCE=2e-5

def discrepancy(x,y):
    if (x is None)!=(y is None):raise ValueError('Availability changed')
    if x is None:return 0.
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):raise ValueError('Nonfinite value')
    return abs(float(x)-float(y))

def accept_report(report):
    if report['tested_patients']!=report['expected_patients']:raise ValueError('Incomplete invariance census')
    if report['comparisons']==0 or report['coverage_comparisons']==0:raise ValueError('Empty invariance checks')
    for field in ['probability_max_delta','coverage_max_delta']:
        if report[field]>TOLERANCE:raise ValueError('TabR invariance failed: '+field)
    if report['availability_changes']:raise ValueError('Availability changed')
    return dict(report,status='passed',absolute_tolerance=TOLERANCE)
