"""Synthetic geometry and scoring checks; no evaluator or neural fitting."""
import json
import math
import numpy as np
import pytest
from tools.probe_structure_codebook_scale import FrozenCodebook, codebook_factor, score
from ra_lsbo.risk_core.observations import OutputObservation

VARS=[{'name':'layout','kind':'nominal','values':['off','on'],'default':'off','active_when':{}},
      {'name':'x','kind':'continuous','bounds':[2.,6.],'default':4.,'active_when':{}},
      {'name':'y','kind':'continuous','bounds':[0.,10.],'default':3.,'active_when':{'layout':['on']}}]
CS=[{'layout':'off'},{'layout':'on'}]
BOOK=np.array([[0.,0.],[2.,4.]])


def test_legacy_exact():
    f=FrozenCodebook(VARS,CS,BOOK)
    assert np.array_equal(f.transform([{'discrete':CS[0],'continuous':{'x':4.,'y':3.}}]),[[.5,0.,1.,0.,0.,0.]])


def test_active_exact():
    f=FrozenCodebook(VARS,CS,BOOK)
    assert np.array_equal(f.transform([{'discrete':CS[1],'continuous':{'x':6.,'y':2.}}]),[[1.,.2,1.,1.,2.,4.]])


def test_no_label_argument():
    f=FrozenCodebook(VARS,CS,BOOK,'unit_pair_median')
    assert f.factor==math.sqrt(10.)


def test_positive_scale_invariance():
    a=BOOK/codebook_factor(BOOK,'unit_pair_median')
    b=(17*BOOK)/codebook_factor(17*BOOK,'unit_pair_median')
    assert np.allclose(a,b)


def test_translation_preserves_distances():
    assert codebook_factor(BOOK+17.,'unit_pair_median')==codebook_factor(BOOK,'unit_pair_median')


def test_median_one():
    b=np.array([[0.,0.],[1.,0.],[1.,2.]])
    b=b/codebook_factor(b,'unit_pair_median')
    d=np.sqrt(np.mean((b[:,None]-b[None])**2,axis=-1))
    assert np.median(d[np.triu_indices(3,1)])==pytest.approx(1.)


@pytest.mark.parametrize('book',[np.zeros((2,2)),np.array([[0.,0.],[np.nan,1.]]),np.array([1.,2.]),np.ones((1,2))])
def test_bad_codebook(book):
    with pytest.raises(ValueError): codebook_factor(book,'unit_pair_median')


def test_unknown_metric():
    with pytest.raises(ValueError): codebook_factor(BOOK,'tuned')


def test_preserve_continuous_and_mask():
    c=[{'discrete':CS[1],'continuous':{'x':6.,'y':2.}}]
    a=FrozenCodebook(VARS,CS,BOOK).transform(c);b=FrozenCodebook(VARS,CS,BOOK,'unit_pair_median').transform(c)
    assert np.array_equal(a[:,:4],b[:,:4])


@pytest.mark.parametrize('c',[
    {'discrete':{'layout':'bad'},'continuous':{'x':4.,'y':3.}},
    {'discrete':CS[0],'continuous':{'x':4.,'y':2.}},
    {'discrete':CS[1],'continuous':{'x':7.,'y':2.}},
    {'discrete':CS[1],'continuous':{'x':float('nan'),'y':2.}},
    {'discrete':CS[1],'continuous':{'x':4.}},
])
def test_invalid_candidates(c):
    with pytest.raises(ValueError): FrozenCodebook(VARS,CS,BOOK).transform([c])


def test_no_mutation():
    b=BOOK.copy();v=json.loads(json.dumps(VARS));f=FrozenCodebook(v,CS,b)
    b[:]=100.;v[0]['default']='on'
    assert np.array_equal(f.codebook,BOOK) and f.variables[0]['default']=='off'


def test_exact_score():
    r=score(0.,1.,0.,OutputObservation('exact','e',value=0.),'soc')
    assert r['nll']==pytest.approx(.5*math.log(2*math.pi)) and r['brier']==.25


def test_interval_score():
    r=score(0.,1.,0.,OutputObservation('interval','e',lower=0.,lower_inclusive=False),'soc')
    assert r['nll']==pytest.approx(math.log(2)) and r['brier']==.25
    assert 'squared_error' not in r


def test_zero_inclusive_unresolved():
    r=score(0.,1.,0.,OutputObservation('interval','e',lower=0.),'soc')
    assert 'brier' not in r


def test_no_missing_imputation():
    with pytest.raises(ValueError): score(0.,1.,0.,OutputObservation('missing','e',reason='unknown'),'soc')


def test_duplicate_identity():
    with pytest.raises(ValueError):FrozenCodebook(VARS,[CS[0],CS[0]],BOOK)
