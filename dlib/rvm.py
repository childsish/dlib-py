import _rvm

class RadialBasisKernel(_rvm.radial_basis_kernel):
    pass

class PolynomialKernel(_rvm.polynomial_kernel):
    pass

class SigmoidKernel(_rvm.sigmoid_kernel):
    pass

class LinearKernel(_rvm.linear_kernel):
    pass

class Trainer(object):
    def __init__(self, kernel, epsilon=0.001):
        self.trainer = self.__getTrainer(kernel)
        self.trainer.set_epsilon(epsilon)
    
    def getEpsilon(self):
        return self.trainer.get_epsilon()
    
    def setEpsilon(self, epsilon):
        self.trainer.set_epsilon(epsilon)
    
    def getKernel(self):
        return self.trainer.get_kernel()
    
    def setKernel(self, kernel):
        tmp = self.trainer.get_epsilon()
        self.trainer = self.__getTrainer(kernel)
        self.trainer.set_epsilon(tmp)
    
    def train(self, samples, labels):
        res = DecisionFunction()
        res.fn = self.trainer.train(VectorSample(samples), VectorDouble(labels))
        return res
    
    def trainProbabilistic(self, samples, labels, folds=3):
        res = ProbabilisticFunction()
        res.fn = _rvm.train_probabilistic_decision_function(self.trainer,
         VectorSample(samples), VectorDouble(labels), folds)
        return res
    
    def __getTrainer(self, kernel):
        if issubclass(type(kernel), _rvm.radial_basis_kernel):
            res = _rvm.rvm_trainer_rbf()
        elif issubclass(type(kernel), _rvm.polynomial_kernel):
            res = _rvm.rvm_trainer_ply()
        elif issubclass(type(kernel), _rvm.sigmoid_kernel):
            res = _rvm.rvm_trainer_sig()
        elif issubclass(type(kernel), _rvm.linear_kernel):
            res = _rvm.rvm_trainer_lin()
        else:
            raise TypeError('Unkown kernel type: %s'%(str(type(kernel))))
        res.set_kernel(kernel)
        return res

class RegressionTrainer(object):
    def __init__(self, kernel, epsilon=0.001):
        self.trainer = self.__getTrainer(kernel)
        self.trainer.set_epsilon(epsilon)
    
    def getEpsilon(self):
        return self.trainer.get_epsilon()
    
    def setEpsilon(self, epsilon):
        self.trainer.set_epsilon(epsilon)
    
    def getKernel(self):
        return self.trainer.get_kernel()
    
    def setKernel(self, kernel):
        tmp = self.trainer.get_epsilon()
        self.trainer = self.__getTrainer(kernel)
        self.trainer.set_epsilon(tmp)
    
    def train(self, samples, labels):
        res = DecisionFunction()
        res.fn = self.trainer.train(VectorSample(samples), VectorDouble(labels))
        return res
    
    def trainProbabilistic(samples, labels, folds=3):
        res = ProbabilisticFunction()
        res.fn = _rvm.train_probabilistic_decision_function(self.trainer,
         VectorSample(samples), VectorDouble(labels), folds)
        return res
    
    def __getTrainer(self, kernel):
        if issubclass(type(kernel), _rvm.radial_basis_kernel):
            res = _rvm.rvm_regression_trainer_rbf()
        elif issubclass(type(kernel), _rvm.polynomial_kernel):
            res = _rvm.rvm_regression_trainer_ply()
        elif issubclass(type(kernel), _rvm.sigmoid_kernel):
            res = _rvm.rvm_regression_trainer_sig()
        elif issubclass(type(kernel), _rvm.linear_kernel):
            res = _rvm.rvm_regression_trainer_lin()
        else:
            raise TypeError('Unknown kernel type: %s'%str(type(kernel)))
        res.set_kernel(kernel)
        return res

class VectorNormalizer(object):
    def __init__(self):
        self.normalizer = _rvm.vector_normalizer()
    
    def __call__(self, sample):
        res = self.normalizer(Sample(sample))
        return [res[i] for i in xrange(len(res))]
    
    def train(self, samples):
        self.normalizer.train(VectorSample(samples))

class DecisionFunction(object):
    def __init__(self):
        self.fn = None
    
    def __call__(self, val):
        return self.fn(Sample(val))
    
    #TODO: Implement pickling later
    def serialize(self, fname):
        _rvm.serialize(self.fn, fname)
    
    #TODO: Implement pickling later
    @classmethod
    def deserialize(cls, fname):
        for rvm_cls in (_rvm.decision_function_rbf,
         _rvm.decision_function_ply,
         _rvm.decision_function_sig,
         _rvm.decision_function_lin):
            try:
                fn = rvm_cls()
                _rvm.deserialize(fn, fname)
                res = DecisionFunction()
                res.fn = fn
                return res
            except RuntimeError:
                continue
        raise RuntimeError('Unable to deserialize file. Unknown function type.')

class ProbabilisticFunction(object):
    def __init__(self):
        self.fn = None
    
    def __call__(self, val):
        return self.fn(Sample(val))
    
    #TODO: Implement pickling later
    def serialize(self, fname):
        _rvm.serialize(self.ndf, fname)
    
    #TODO: Implement pickling later
    @classmethod
    def deserialize(cls, fname):
        for rvm_cls in (_rvm.probabilistic_function_rbf,
         _rvm.probabilistic_function_ply,
         _rvm.probabilistic_function_sig,
         _rvm.probabilistic_function_lin):
            try:
                fn = rvm_cls()
                _rvm.deserialize(fn, fname)
                res = ProbabilisticFunction()
                res.fn = fn
                return res
            except RuntimeError:
                continue
        raise RuntimeError('Unable to deserialize file. Unknown function type.')

class NormalizedFunction(object):
    def __init__(self, function, normalizer=VectorNormalizer()):
        """ Expects a python function (eg. DecisionFunction) and a python normalizer
         (eg.) VectorNormalizer. """
        self.fn = self.__getNormalizedFunction(function.fn)
        self.fn.normalizer = normalizer.normalizer
    
    def __call__(self, val):
        return self.fn(Sample(val))
    
    def getNormalizer(self):
        return self.fn.normalizer
    
    def setNormalizer(self, normalizer):
        self.fn.normalizer = normalizer
    
    def getFunction(self):
        return self.fn.function
    
    def setFunction(self, function):
        tmp = self.fn.normalizer
        self.fn = self.__getNormalizedFunction(function)
        self.fn.normalizer = tmp
    
    #TODO: Implement pickling later
    def serialize(self, fname):
        _rvm.serialize(self.fn, fname)
    
    #TODO: Implement pickling later
    @classmethod
    def deserialize(cls, fname):
        for rvm_cls in (_rvm.normalized_decision_function_rbf,
         _rvm.normalized_decision_function_ply,
         _rvm.normalized_decision_function_sig,
         _rvm.normalized_decision_function_lin,
         _rvm.normalized_probabilistic_function_rbf,
         _rvm.normalized_probabilistic_function_ply,
         _rvm.normalized_probabilistic_function_sig,
         _rvm.normalized_probabilistic_function_lin):
            try:
                fn = rvm_cls()
                _rvm.deserialize(fn, fname)
                # Create a temporary DecisionFunction because the NormalizedFunction
                #  constructor expects a Python object
                tmp = DecisionFunction()
                tmp.fn = fn.function
                res = NormalizedFunction(tmp)
                res.fn.normalizer = fn.normalizer
                return res
            except RuntimeError:
                continue
        raise RuntimeError('Unable to deserialize file. Unknown function type.')
    
    def __getNormalizedFunction(self, function):
        if issubclass(type(function), _rvm.decision_function_rbf):
            res = _rvm.normalized_decision_function_rbf()
        elif issubclass(type(function), _rvm.decision_function_ply):
            res = _rvm.normalized_decision_function_ply()
        elif issubclass(type(function), _rvm.decision_function_sig):
            res = _rvm.normalized_decision_function_sig()
        elif issubclass(type(function), _rvm.decision_function_lin):
            res = _rvm.normalized_decision_function_lin()
        elif issubclass(type(function), _rvm.probabilistic_function_rbf):
            res = _rvm.normalized_probabilistic_function_rbf()
        elif issubclass(type(function), _rvm.probabilistic_function_ply):
            res = _rvm.normalized_probabilistic_function_ply()
        elif issubclass(type(function), _rvm.probabilistic_function_sig):
            res = _rvm.normalized_probabilistic_function_sig()
        elif issubclass(type(function), _rvm.probabilistic_function_lin):
            res = _rvm.normalized_probabilistic_function_lin()
        else:
            raise TypeError('Unknown function type: %s'%str(type(function)))
        res.function = function
        return res

def VectorSample(x):
    res = _rvm.vector_sample(len(x))
    for i in xrange(len(x)):
        res[i] = Sample(x[i])
    return res

def VectorDouble(x):
    res = _rvm.vector_double()
    for i in xrange(len(x)):
        res.append(x[i])
    return res

def Sample(x):
    res = _rvm.sample(len(x))
    for i in xrange(len(x)):
        res[i] = x[i]
    return res
