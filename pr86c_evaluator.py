"""PR86-C pure analytic simulator interface; reserved tasks are FULL-only.

Learning/acquisition modules must never import or call truth_for_evaluation.
"""
import math

from risk_learning.core import TASKS as RELEASED, canonical, truth as released_truth
from risk_learning.core import tradeoff_geometry
from pr86c_protocol import HELDOUT


def truth_for_evaluation(task, candidate):
    if task in RELEASED:
        return released_truth(task, candidate)
    if task not in HELDOUT:
        raise ValueError("Task is not released in PR86-C")
    c = canonical(candidate)
    b, x, y, t = (c[k] for k in ("branch", "x", "y", "trim"))
    s = int(task[-1])
    cx, cy, rx, ry, angle = tradeoff_geometry(s,b,t)
    dx,dy = x-cx,y-cy
    u = (math.cos(angle)*dx+math.sin(angle)*dy)/rx
    v = (-math.sin(angle)*dx+math.cos(angle)*dy)/ry
    region = u*u+v*v-1.0
    resource = t+0.08*b+0.06*(x+y)-0.90 if b else -0.2
    objective = 0.7*(b-s)**2+(u-0.8)**2+0.7*v*v+(
        0.1*(t-0.35)**2 if b else 0.0)
    return dict(region=region, resource=resource, objective=objective,
                feasible=(region<=0 and resource<=0))
