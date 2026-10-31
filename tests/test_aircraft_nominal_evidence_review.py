import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('review', Path(__file__).parents[1]/'tools/review_aircraft_nominal_evidence.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


@pytest.mark.parametrize('value,label', [
    ({'kind':'exact','value':-1}, True), ({'kind':'exact','value':0}, True),
    ({'kind':'exact','value':1}, False), ({'kind':'missing'}, None),
    ({'kind':'interval','lower':0,'upper':None,'lower_inclusive':False}, False),
    ({'kind':'interval','lower':0,'upper':None,'lower_inclusive':True}, None),
    ({'kind':'interval','lower':.1,'upper':None,'lower_inclusive':True}, False),
    ({'kind':'interval','lower':None,'upper':0,'lower_inclusive':True}, True),
    ({'kind':'interval','lower':-1,'upper':1,'lower_inclusive':True}, None),
])
def test_partial_observation_meaning(value, label):
    assert r.satisfaction(value) is label


@pytest.mark.parametrize('status,kind,margin,certified,expected', [
    ('completed','exact',-.1,False,True), ('completed','exact',.1,False,False),
    ('unknown','exact',-.1,False,False), ('infrastructure_error','exact',-.1,False,False),
    ('completed','interval',-.1,False,False), ('completed','exact',-.1,True,False),
])
def test_usable(status, kind, margin, certified, expected):
    row = {'observation':{'status':status,'objective':{'kind':kind},
                         'constraints':[['c',{'kind':'exact','value':margin}]]},
           'diagnostics':{'certified_physical_infeasible':certified}}
    assert r.usable(row) is expected


def test_paired_not_difference_of_marginal_medians():
    p = r.pairing({1:3,2:11,3:20}, {1:4,2:1,3:20})
    assert p['left_better'] == p['left_worse'] == p['ties'] == 1
    assert p['median_difference_Wh'] == 0


def test_pairing_refuses_missing_seed():
    with pytest.raises(ValueError):
        r.pairing({1:1}, {2:1})


def test_hash_does_not_change_tuple_serialization():
    assert r.digest({'a':((1,2),)}) == r.digest({'a':[[1,2]]})


def test_unknown_cannot_be_feasible():
    assert not r.usable({'observation':{'status':'completed','objective':{'kind':'exact'},
                                       'constraints':[['c',{'kind':'missing'}]]},'diagnostics':{}})


def test_corrupt_archive_stops_before_parsing(tmp_path):
    f = tmp_path/'bad'; f.write_bytes(b'not a zip')
    with pytest.raises(ValueError, match='Wrong raw archive'):
        r.review(f, f)
