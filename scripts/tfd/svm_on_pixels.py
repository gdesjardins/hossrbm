import numpy
import theano
import pickle
import logging
import time
import copy
from pylearn2.datasets import tfd
from sklearn.svm import LinearSVC
from pipelines.svm.svm_on_features import cross_validate_svm, retrain_svm, compute_test_error, process_labels

logging.basicConfig(level=logging.INFO)

class Standardize():

    def __init__(self, X, eps=1e-2):
        self.mean = X.mean(axis=0)
        self.std = eps + X.std(axis=0)

    def apply(self, X):
        return (X - self.mean) / self.std

def run_fold(fold_i, C_list=[1.0]):
    train = tfd.TFD('train', fold=fold_i)
    valid = tfd.TFD('valid', fold=fold_i)
    test  = tfd.TFD('test', fold=fold_i)

    preproc = Standardize(train.X / 255.)
    train.X = preproc.apply(train.X / 255.)
    valid.X = preproc.apply(valid.X / 255.)
    test.X  = preproc.apply(test.X / 255.)
    train_y = process_labels(train.y)
    valid_y = process_labels(valid.y)
    test_y = process_labels(test.y)

    svm = LinearSVC(C=1.0, loss='l2', penalty='l2')
    (svm, valid_error) = cross_validate_svm(svm, (train.X, train_y), (valid.X, valid_y), C_list)
    svm = retrain_svm(svm, (train.X, train_y), (valid.X, valid_y))
    test_error = compute_test_error(svm, (test.X, test_y))
    print 'Fold %i: valid_error = %f\t  test_error = %f' % (fold_i, valid_error, test_error)
    return (valid_error, test_error, svm.C)

valerrs = []
tsterrs = []
for i in xrange(5):
    C_list = [1e-4,1e-3,1e-2,0.1,1.0,10,100] if i==0 else [bestC]
    valerr, tsterr, bestC = run_fold(i, C_list)
    valerrs += [valerr]
    tsterrs += [tsterr]

print 'Average validation error: ', numpy.array(valerrs).mean()
print 'Average test error: ', numpy.array(tsterrs).mean()
