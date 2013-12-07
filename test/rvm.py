'''
Created on 28.03.2013

@author: Liam
'''

import math

from dlib import rvm
from unittest import TestCase

class TestVectorNormaliser(TestCase):
    def test_train(self):
        pass

def sinc(x):
    if x == 0:
        return 1
    return math.sin(x) / x

def test_rvc():
    samples = []
    labels = []
    
    for r in xrange(-20, 20):
        for c in xrange(-20, 20):
            s = (float(r), float(c))
            samples.append(s)
            
            if (math.sqrt(r*r + c*c) <= 10):
                labels.append(1.)
            else:
                labels.append(-1.)
    
    normalizer = VectorNormalizer()
    normalizer.train(samples)
    for i in xrange(len(samples)):
        samples[i] = normalizer(samples[i])
    
    trainer = Trainer(RadialBasisKernel(0.08), 0.01)
    
    fn = NormalizedFunction(trainer.train(samples, labels), normalizer)
    
    for x, y in ((3.123, 2), (3.123, 9.3545)):
        print "This sample should be >= 0 and it is classified as a %.3f"%fn((x, y))

    for x, y in ((13.123, 9.3545), (13.123, 0)):
        print "This sample should be < 0 and it is classified as a %.3f"%fn((x, y))
    
    fn.serialize("saved_function.dat")
    fn = NormalizedFunction.deserialize("saved_function.dat")
    print fn((x, y))
    
    pfn = NormalizedFunction(trainer.trainProbabilistic(samples, labels), normalizer)
    
    for x, y in ((3.123, 2), (3.123, 9.3545)):
        print "This +1 example should have high probability. Its probability is: %.3f"%pfn((x, y))
    
    for x, y in ((13.123, 9.3545), (13.123, 0)):
        print "This -1 example should have low probability. Its probability is: %.3f"%pfn((x, y))
    
    pfn.serialize("saved_function.dat")
    pfn = NormalizedFunction.deserialize("saved_function.dat")
    print pfn(s)

def test_rvr():
    samples = []
    labels = []
    
    for x in xrange(-10, 4):
        samples.append((x,))
        labels.append(sinc(x))
    
    gamma = 2.0 / rvm_binding.compute_mean_squared_distance(VectorSample(samples))
    trainer = RegressionTrainer(RadialBasisKernel(gamma), 0.00001)
    print 'using gamma of', gamma
    
    fn = trainer.train(samples, labels)
    
    m = (2.5,); print sinc(m[0]), fn(m)
    m = (0.1,); print sinc(m[0]), fn(m)
    m = (-4.,); print sinc(m[0]), fn(m)
    m = (5.0,); print sinc(m[0]), fn(m)
    
    fn.serialize("saved_function.dat")
    fn = DecisionFunction.deserialize("saved_function.dat")
    print fn(m)

def main():
    test_rvc()
    test_rvr()
