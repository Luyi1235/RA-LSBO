from copy import deepcopy
import pytest
from ra_lsbo.experiments.simplified_aircraft import corpus_protocol as p

@pytest.mark.parametrize('name,cap,n', [('smoke',42,10),('pilot',2672,80)])
def test_budget(name,cap,n):
    b=p.budget_for(p.profile_for(name))
    assert b['maximum_model_calls']==cap
    assert b['incumbent_step_verification']==n
    assert b['maximum_rcaide_mda_calls']==0
    assert b['shared_initial']+b['online']+n==cap

@pytest.mark.parametrize('name',['confirm','resume','mda','transfer','full'])
def test_no_other_stages(name):
    with pytest.raises(ValueError):p.profile_for(name)

@pytest.mark.parametrize('method',['original_lsbo','corpus_reset','ev_equal','ev_prequential','bad'])
def test_no_extra_arms(method):
    with pytest.raises(ValueError):p.method_spec(method,p.profile_for('pilot'))

def test_new_fixed_seeds_and_no_alias_mutation():
    x=p.profile_for('pilot');assert x['seeds']==list(range(9931,9939))
    x['seeds'][0]=1;assert p.profile_for('pilot')['seeds'][0]==9931

def test_unchanged_search_and_kernel_except_declared_representation():
    profile=p.profile_for('pilot');a=p.method_spec('old30',profile)
    for arm in ('old480','corpus_persistent','onehot'):
        b=p.method_spec(arm,profile)
        x=deepcopy(a);y=deepcopy(b)
        x.pop('representation_policy');y.pop('representation_policy')
        x['settings'].pop('encoder_epochs');y['settings'].pop('encoder_epochs')
        x['settings'].pop('latent');y['settings'].pop('latent')
        assert x==y
    assert p.model_settings('old480',profile).encoder_epochs==480
    assert p.model_settings('onehot',profile).latent is False

def test_smoke_reduces_search_not_corpus_training():
    x=p.method_spec('corpus_persistent',p.profile_for('smoke'))
    y=p.method_spec('corpus_persistent',p.profile_for('pilot'))
    assert x['representation_policy']==y['representation_policy']
    assert x['settings']['encoder_epochs']==30

def test_orders_are_outcome_independent_complete_and_cyclic():
    profile=p.profile_for('pilot')
    for case in p.CASES:
        for seed in profile['seeds']:
            order=p.method_order(profile,case,seed)
            assert len(order)==5 and set(order)==set(p.ARMS)
            assert order==p.method_order(profile,case,seed)
    assert p.method_order(profile,p.CASES[0],9931)!=p.method_order(profile,p.CASES[0],9932)
