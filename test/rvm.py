import math
import itertools
import unittest
import tempfile

from dlib._rvm import compute_mean_squared_distance as msd
from dlib.rvm import *

class TestRVMs(unittest.TestCase):
    def setUp(self):
        self.c_samples = [(float(r), float(c)) for r, c in\
            itertools.product(xrange(-20, 20), xrange(-20, 20))]
        self.c_labels = [1 if math.sqrt(r*r + c*c) <= 10 else -1\
            for r, c in self.c_samples]
        
        self.r_samples = [(x,) for x in range(-10, 4)]
        self.r_labels = [sinc(x) for (x,) in self.r_samples]
        
        self.fhndl, self.fname = tempfile.mkstemp()

    def test_rvc(self):
        labels = self.c_labels
        normalizer = VectorNormalizer()
        normalizer.train(self.c_samples)
        samples = [normalizer(sample) for sample in self.c_samples]
        trainer = Trainer(RadialBasisKernel(0.08), 0.01)
        fn = NormalizedFunction(trainer.train(samples, labels), normalizer)

        self.assertGreaterEqual(fn((3.123, 2)), 0)
        self.assertGreaterEqual(fn((3.123, 9.3545)), 0)
        self.assertLess(fn((13.123, 9.3545)), 0)
        self.assertLess(fn((13.123, 0)), 0)
        
    def test_probabilistic_rvc(self):
        labels = self.c_labels
        normalizer = VectorNormalizer()
        normalizer.train(self.c_samples)
        samples = [normalizer(sample) for sample in self.c_samples]
        trainer = Trainer(RadialBasisKernel(0.08), 0.01)
        pfn = NormalizedFunction(trainer.trainProbabilistic(samples, labels),
            normalizer)
        
        self.assertGreaterEqual(pfn((3.123, 2)), 0.5)
        self.assertGreaterEqual(pfn((3.123, 9.3545)), 0.5)
        self.assertLess(pfn((13.123, 9.3545)), 0.5)
        self.assertLess(pfn((13.123, 0)), 0.5)
    
    def test_serialize_rvc(self):
        labels = self.c_labels
        normalizer = VectorNormalizer()
        normalizer.train(self.c_samples)
        samples = [normalizer(sample) for sample in self.c_samples]
        trainer = Trainer(RadialBasisKernel(0.08), 0.01)
        fn = NormalizedFunction(trainer.train(samples, labels), normalizer)
        fn.serialize(self.fname)
        fn = NormalizedFunction.deserialize(self.fname)

        self.assertGreaterEqual(fn((3.123, 2)), 0)
        self.assertGreaterEqual(fn((3.123, 9.3545)), 0)
        self.assertLess(fn((13.123, 9.3545)), 0)
        self.assertLess(fn((13.123, 0)), 0)
        
    def test_serialize_probabilistic_rvc(self):
        labels = self.c_labels
        normalizer = VectorNormalizer()
        normalizer.train(self.c_samples)
        samples = [normalizer(sample) for sample in self.c_samples]
        trainer = Trainer(RadialBasisKernel(0.08), 0.01)
        fn = NormalizedFunction(trainer.trainProbabilistic(samples, labels),
            normalizer)
        fn.serialize(self.fname)
        fn = NormalizedFunction.deserialize(self.fname)

        self.assertGreaterEqual(fn((3.123, 2)), 0.5)
        self.assertGreaterEqual(fn((3.123, 9.3545)), 0.5)
        self.assertLess(fn((13.123, 9.3545)), 0.5)
        self.assertLess(fn((13.123, 0)), 0.5)
    
    def test_rvr(self):
        gamma = 2.0 / msd(VectorSample(self.r_samples))
        trainer = RegressionTrainer(RadialBasisKernel(gamma), 0.00001)
        fn = trainer.train(self.r_samples, self.r_labels)
        
        self.assertAlmostEqual(fn((2.5,)), 0.2594000)
        self.assertAlmostEqual(fn((0.1,)), 0.9812903)
        self.assertAlmostEqual(fn((-4.,)), -0.1188682)
        self.assertAlmostEqual(fn((5.0,)), -0.69175494)
    
    def test_serialize_rvr(self):
        gamma = 2.0 / msd(VectorSample(self.r_samples))
        trainer = RegressionTrainer(RadialBasisKernel(gamma), 0.00001)
        fn = trainer.train(self.r_samples, self.r_labels)
        fn.serialize(self.fname)
        fn = DecisionFunction.deserialize(self.fname)
        
        self.assertAlmostEqual(fn((2.5,)), 0.2594000)
        self.assertAlmostEqual(fn((0.1,)), 0.9812903)
        self.assertAlmostEqual(fn((-4.,)), -0.1188682)
        self.assertAlmostEqual(fn((5.0,)), -0.69175494)
    
    def test_basisVectors(self):
        labels = self.c_labels
        normalizer = VectorNormalizer()
        normalizer.train(self.c_samples)
        samples = [normalizer(sample) for sample in self.c_samples]
        trainer = Trainer(RadialBasisKernel(0.08), 0.01)
        fn = NormalizedFunction(trainer.train(samples, labels), normalizer)
        vectors = fn.basis_vectors
    
        self.assertEquals(len(vectors), 10)
        self.assertAlmostEquals(vectors[0][0], -1.6887495)
        self.assertAlmostEquals(vectors[0][1], -1.6887495)
        self.assertAlmostEquals(vectors[1][0], -1.6887495)
        self.assertAlmostEquals(vectors[1][1], 1.6887495)
        self.assertAlmostEquals(vectors[2][0], -0.8227241)
        self.assertAlmostEquals(vectors[2][1], -1.2557368)
        self.assertAlmostEquals(vectors[3][0], -0.4763140)
        self.assertAlmostEquals(vectors[3][1], -0.5629165)
        self.assertAlmostEquals(vectors[4][0], -0.4763140)
        self.assertAlmostEquals(vectors[4][1], 0.3031089)
        self.assertAlmostEquals(vectors[5][0], 0.1299038)
        self.assertAlmostEquals(vectors[5][1], 0.1299038)
        self.assertAlmostEquals(vectors[6][0], 0.3897114)
        self.assertAlmostEquals(vectors[6][1], -0.0433013)
        self.assertAlmostEquals(vectors[7][0], 1.2557368)
        self.assertAlmostEquals(vectors[7][1], 0.7361216)
        self.assertAlmostEquals(vectors[8][0], 1.5155445)
        self.assertAlmostEquals(vectors[8][1], 1.5155445)
        self.assertAlmostEquals(vectors[9][0], 1.6887495)
        self.assertAlmostEquals(vectors[9][1], -1.6887495)

def sinc(x):
    if x == 0:
        return 1
    return math.sin(x) / x

if __name__ == '__main__':
    import sys
    sys.exit(unittest.main())
