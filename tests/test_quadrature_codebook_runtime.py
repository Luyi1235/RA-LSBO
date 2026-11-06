"""Actual native geometry with synthetic archived codebooks; no fits/evaluations.

These unit checks do not replace the four-state historical checkpoint CI.
"""
from types import SimpleNamespace
import unittest
import numpy as np
from ra_lsbo.budget import digest
from ra_lsbo.experiments.risk_budget63_runner import _case
from quadrature_checkpoint_features import ArchivedCodebookFeatures


class ArchivedGeometryTests(unittest.TestCase):
    def make(self):
        p = _case('branin_sparse')
        f = ArchivedCodebookFeatures.__new__(ArchivedCodebookFeatures)
        f.space = p.space
        f.modules = {'space': SimpleNamespace(Candidate=p.candidate_class)}
        configs = p.space.configurations()
        f._archived_codebook = {digest(c): [float(i), float(i+1), float(i+2), float(i+3)]
                               for i,c in enumerate(configs)}
        candidates = [p.canonical({'continuous': {'x': .2, 'y': .3, 'trim': .8}, 'discrete': c})
                      for c in configs]
        return p,f,candidates

    def test_each_configuration_uses_its_sealed_vector(self):
        _,f,cs = self.make()
        raw=f._raw(cs)
        self.assertEqual(raw.shape,(4,10))
        for c,row in zip(cs,raw):
            np.testing.assert_array_equal(row[-4:], f._archived_codebook[digest(c['discrete'])])

    def test_active_geometry_is_the_frozen_native_geometry(self):
        p,f,cs = self.make();raw=f._raw(cs)
        for c,row in zip(cs,raw):
            native=p.candidate_class(**c)
            mask=np.asarray(p.space.mask(native.discrete))
            np.testing.assert_array_equal(row[:3], np.asarray(p.space.unit(native))*mask)
            np.testing.assert_array_equal(row[3:6],mask)

    def test_batch_order_and_repetition_cannot_change_latents(self):
        _,f,cs=self.make();one=f._raw(cs)
        many=f._raw([cs[3],cs[1],cs[1],cs[0]])
        np.testing.assert_array_equal(many,one[[3,1,1,0]])

    def test_missing_archived_configuration_is_not_invented(self):
        _,f,cs=self.make()
        del f._archived_codebook[digest(cs[0]['discrete'])]
        with self.assertRaises(KeyError):f._raw([cs[0]])


if __name__=='__main__':unittest.main()
