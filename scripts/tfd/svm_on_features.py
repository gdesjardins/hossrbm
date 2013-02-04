import numpy
import theano
import pickle
import logging
import time
import copy
import argparse
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

def run_fold(path, fold_i, C_list=[1.0]):
    train = pickle.load(open('%s/fold%i/train.pkl' % (path, fold_i)))
    valid = pickle.load(open('%s/fold%i/valid.pkl' % (path, fold_i)))
    test  = pickle.load(open('%s/fold%i/test.pkl' % (path, fold_i)))

    preproc = Standardize(train.X)
    train.X = preproc.apply(train.X)
    valid.X = preproc.apply(valid.X)
    test.X  = preproc.apply(test.X)
    train_y = process_labels(train.y)
    valid_y = process_labels(valid.y)
    test_y = process_labels(test.y)

    svm = LinearSVC(C=1.0, loss='l2', penalty='l2')
    if len(C_list) == 1:
        svm.set_params(C = C_list[0])
        svm = retrain_svm(svm, (train.X, train_y), (valid.X, valid_y))
        valid_error = -1.
    else:
        (svm, valid_error) = cross_validate_svm(svm, (train.X, train_y), (valid.X, valid_y), C_list)
        svm = retrain_svm(svm, (train.X, train_y), (valid.X, valid_y))
    test_error = compute_test_error(svm, (test.X, test_y))
    print 'Fold %i: valid_error = %f\t  test_error = %f' % (fold_i, valid_error, test_error)
    return (valid_error, test_error, svm.C)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to directory containing first layer features.')
    parser.add_argument('--C', default='0.0001,0.001,0.01,0.1,1.')
    args = parser.parse_args()

    C_list = numpy.array(args.C.split(','), dtype='float32')
    valerrs = []
    tsterrs = []
    for i in xrange(5):
        C_list = C_list if i==0 else [bestC]
        print '#### Running fold %i ####' % i
        valerr, tsterr, bestC = run_fold(args.path, i, C_list)
        valerrs += [valerr]
        tsterrs += [tsterr]

    print 'Average validation error: ', numpy.array(valerrs).mean()
    print 'Average test error: ', numpy.array(tsterrs).mean()
