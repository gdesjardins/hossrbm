import numpy
from pylearn2.datasets import dense_design_matrix as ddm

class DenseDesignMatrixMixture(ddm.DenseDesignMatrix):

    def __init__(self, datasets, pvals, rng=None):
        self.datasets = datasets
        self.pvals = pvals
        self.rng = rng if rng else numpy.random.RandomState(93827104)

    def get_batch_design(self, batch_size, include_labels=False):
        idx = self.rng.multinomial(1, self.pvals)
        idx = numpy.where(idx == 1)[0]
        return self.datasets[idx].get_batch_design(batch_size, include_labels) 
