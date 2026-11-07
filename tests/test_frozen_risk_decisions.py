import itertools
import math
import unittest
from decision_value_core import (BUDGETS, MODELS, TASKS, SIZES, aggregate, average,
                                 digest, fixed_plan, random_reference, rank_candidates, score_prefix)


def ids(n):
    return [digest(['test', i]) for i in range(n)]


class FrozenDecisionTests(unittest.TestCase):
    def test_probability_order(self):
        self.assertEqual(rank_candidates(ids(3), [.1,.9,.5]), [1,2,0])

    def test_tie_order_invariant_to_input_permutation(self):
        cs = ids(5)
        expected = [cs[i] for i in rank_candidates(cs, [.2]*5)]
        rev = cs[::-1]
        self.assertEqual([rev[i] for i in rank_candidates(rev,[.2]*5)], expected)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError): rank_candidates([ids(1)[0]]*2,[.1,.2])

    def test_invalid_probabilities(self):
        for value in (float('nan'),float('inf'),-.1,1.1,True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rank_candidates(ids(1),[value])

    def test_invalid_id(self):
        with self.assertRaises(ValueError): rank_candidates(['unknown'],[.5])

    def test_alignment(self):
        with self.assertRaises(ValueError): rank_candidates(ids(2),[.5])

    def test_empty(self):
        with self.assertRaises(ValueError): rank_candidates([],[])

    def test_no_labels_in_selector(self):
        import inspect
        self.assertEqual(list(inspect.signature(rank_candidates).parameters),['ids','probabilities'])

    def test_rank_does_not_mutate_inputs(self):
        cs, ps=ids(3),[.5,.3,.6]
        before=(list(cs),list(ps));rank_candidates(cs,ps)
        self.assertEqual((cs,ps),before)

    def test_random_exact_small_population(self):
        labels=[True,True,False,False,False]
        subsets=list(itertools.combinations(range(5),3))
        value=random_reference(5,2,3)
        self.assertAlmostEqual(value['expected_feasible'],sum(sum(labels[i] for i in s) for s in subsets)/len(subsets))
        self.assertAlmostEqual(value['probability_any_feasible'],sum(any(labels[i] for i in s) for s in subsets)/len(subsets))

    def test_random_no_feasible(self):
        self.assertEqual(random_reference(512,0,12)['probability_any_feasible'],0.)

    def test_random_all_feasible(self):
        self.assertEqual(random_reference(512,512,12)['expected_feasible'],12.)
        self.assertEqual(random_reference(512,512,12)['probability_any_feasible'],1.)

    def test_random_full_pool(self):
        self.assertEqual(random_reference(5,1,5)['probability_any_feasible'],1.)

    def test_random_invalid(self):
        for args in [(0,0,1),(4,5,1),(4,1,5),(4,-1,1),(4,1,0),(True,0,1)]:
            with self.subTest(args=args),self.assertRaises(ValueError): random_reference(*args)

    def test_score_first_feasible(self):
        row=score_prefix([2,0,1],[True,False,False],2)
        self.assertEqual(row['first_feasible_rank'],2)
        self.assertEqual(row['feasible_selected'],1)

    def test_score_no_feasible_null(self):
        row=score_prefix([1,0],[True,False],1)
        self.assertIsNone(row['first_feasible_rank']);self.assertFalse(row['any_feasible'])

    def test_score_boolean_contract(self):
        with self.assertRaises(ValueError): score_prefix([0],[1],1)

    def test_score_permutation(self):
        with self.assertRaises(ValueError): score_prefix([0,0],[True,False],1)

    def test_score_budget(self):
        with self.assertRaises(ValueError): score_prefix([0],[True],2)

    def test_new_bill_zero(self):
        self.assertFalse(any(fixed_plan()['new_bill'].values()))

    def test_frozen_plan(self):
        p=fixed_plan()
        self.assertEqual(p['source_model_rows'],360)
        self.assertEqual(p['historical_true_evaluations'],12672)
        self.assertTrue(p['posthoc']);self.assertFalse(p['adaptive_updates'])
        self.assertEqual(BUDGETS,(1,2,4,6,12))

    def test_mean_no_imputation(self):
        self.assertIsNone(average([]));self.assertIsNone(average([1,None]))
        self.assertEqual(average([1.,2.]),1.5)

    def test_roster_missing_rejected(self):
        with self.assertRaises(ValueError): aggregate([])

    def test_frozen_no_hidden_extras(self):
        self.assertEqual(len(TASKS),6);self.assertEqual(len(MODELS),4);self.assertEqual(len(SIZES),5)
        self.assertTrue(all('s4' not in t for t in TASKS))


if __name__ == '__main__':
    unittest.main()
