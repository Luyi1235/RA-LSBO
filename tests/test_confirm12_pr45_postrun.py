"""Zero-aircraft regression tests for PR45 postrun forensic/paired analysis."""
import json
import math
from pathlib import Path
import zipfile

import pytest

from confirm12.postrun_pr45 import (
    CALL_CAP,EXPECTED_GROUPS,EXPECTED_TRIALS,_exact_mcnemar,
    _load_methods,_pair_discovery,_pair_energy,verify_roster,verify_progress,
)
from confirm12.nested_seed_plan import candidate_plan


def _full_plan():
    roster=candidate_plan()
    cases={}
    for case,seeds in roster["seeds_by_case"].items():
        subset=set(roster["auxiliary_by_case"][case])
        cases[case]={
            "seeds":seeds,
            "arms_by_seed":{
                str(s):list(roster["main_arms"])+(
                    list(roster["auxiliary_arms"]) if s in subset else [])
                for s in seeds}}
    return {"max_calls":151040,"workers":12,"cases":cases}


def test_frozen_all_call_arithmetic_and_256_nested_seeds():
    frozen=_full_plan()
    expected=verify_roster(frozen)
    assert len(expected)==EXPECTED_GROUPS==2048
    assert sum(len(v) for v in expected.values())==EXPECTED_TRIALS==6656
    assert CALL_CAP==24576+119808+6656
    assert 151006==24576+119808+6622
    assert CALL_CAP-151006==6656-6622==34


def test_pair_exact_mcnemar_computation_not_asymptotic():
    assert _exact_mcnemar(0,0)==1.
    assert _exact_mcnemar(2,0)==.5
    assert _exact_mcnemar(0,4)==.125
    assert _exact_mcnemar(1,1)==1.


def test_pair_energy_and_discovery_are_seed_matched():
    def item(ok,energy=None):
        return {"nominal_usable_feasible":ok,"nominal_best_energy_Wh":energy}
    fixture={
        ("aircraft_ordinary",7,"ra_lsbo"):item(True,10.),
        ("aircraft_ordinary",7,"full_latent"):item(True,11.),
        ("aircraft_ordinary",8,"ra_lsbo"):item(True,12.),
        ("aircraft_ordinary",8,"full_latent"):item(False),
        ("aircraft_ordinary",9,"ra_lsbo"):item(False),
        ("aircraft_ordinary",9,"full_latent"):item(False),
    }
    e=_pair_energy(fixture,"aircraft_ordinary","full_latent",[7,8,9])
    assert e["paired_feasible"]==1
    assert e["mean_difference_Wh"]==-1.
    d=_pair_discovery(fixture,"aircraft_ordinary","full_latent",[7,8,9])
    assert d["RA_only_feasible"]==1
    assert d["control_only_feasible"]==0
    assert math.isclose(d["paired_rate_difference_pp"],100/3)
    assert d["exact_McNemar_two_sided_p_descriptive"]==1.


def test_zip_loader_detects_immutable_shared_initials(tmp_path):
    archive=tmp_path/"method_results.zip"
    key=("aircraft_ordinary",1)
    expected={key:("ra_lsbo","full_latent")}
    a={
        "case":"aircraft_ordinary","seed":1,"method":"ra_lsbo",
        "online_calls":18,"shared_initial_sha256":"a"*64,
        "nominal_usable_feasible":True,
        "nominal_best_energy_Wh":333.,
        "same_design_5sec_verification":True}
    b={**a,"method":"full_latent","nominal_best_energy_Wh":335.}
    with zipfile.ZipFile(archive,"w") as z:
        for row in [a,b]:
            base=f"any_prefix/trials/aircraft_ordinary/1/{row['method']}"
            z.writestr(base+"/result.json",json.dumps(row))
            z.writestr(base+"/verification.json","{}")
    records,ver,nominal,vcount=_load_methods(tmp_path,expected,archive)
    assert len(records)==len(ver)==2
    assert nominal["aircraft_ordinary","ra_lsbo"]==1
    assert vcount["aircraft_ordinary","full_latent"]==1
    assert _pair_energy(records,"aircraft_ordinary","full_latent",[1])["mean_difference_Wh"]==-2.


def test_progress_rejects_duplicate_or_unregistered_receipts(tmp_path):
    root=tmp_path/"data"
    root.mkdir()
    rows=[
        {"case":"aircraft_ordinary","seed":1,"methods":2,
         "total_groups":2048,"completed_groups":1,"charged_true_calls":50},
        {"case":"aircraft_ordinary","seed":1,"methods":2,
         "total_groups":2048,"completed_groups":2,"charged_true_calls":50},
    ]
    (root/"progress.jsonl").write_text(
        "\n".join(json.dumps(x) for x in rows)+"\n",encoding="utf8")
    review=verify_progress(root,{("aircraft_ordinary",1):("ra_lsbo","full_latent")})
    assert review["errors"]
