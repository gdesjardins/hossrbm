import numpy
import theano
import pickle
import logging
import time
import copy
from pylearn2.datasets import tfd
from sklearn.svm import LinearSVC

logging.basicConfig(level=logging.INFO)

def cross_validate_svm(svm, trainset, validset, C_list):
    best_svm = None
    best_error = numpy.Inf
    for C in C_list:
        t1 = time.time()
        if hasattr(svm, 'set_params'):
            svm.set_params(C = C)
        else:
            svm.C = C
        svm.fit(trainset.X, trainset.y)
        predy = svm.predict(validset.X)
        error = (validset.y != predy).mean()
        if error < best_error:
            logging.info('SVM(C=%f): valid_error=%f **' % (C, error))
            best_error = error
            best_svm = copy.deepcopy(svm)
            # Numpy bug workaround: copy module does not respect C/F ordering.
            best_svm.raw_coef_ = numpy.asarray(best_svm.raw_coef_, order='F')
        else:
            logging.info('SVM(C=%f): valid_error=%f' % (C, error))
        logging.info('Elapsed time: %f' % (time.time() - t1))
    return (best_svm, best_error)

def retrain_svm(svm, trainset, validset):
    assert validset is not None
    logging.info('Retraining on {train, validation} sets.')
    full_train_X = numpy.vstack((trainset.X, validset.X))
    full_train_y = numpy.hstack((trainset.y, validset.y))
    svm.fit(full_train_X, full_train_y)
    return svm

def compute_test_error(svm, testset):
    test_predy = svm.predict(testset.X)
    return (testset.y != test_predy).mean()

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

    svm = LinearSVC(C=1.0, loss='l2', penalty='l2')
    (svm, valid_error) = cross_validate_svm(svm, train, valid, C_list)
    svm = retrain_svm(svm, train, valid)
    test_error = compute_test_error(svm, test)
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
