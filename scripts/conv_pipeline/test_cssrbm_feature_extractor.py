import pickle
import numpy
import theano
import theano.tensor as T

import cssrbm_feature_extractor
from hossrbm import pooled_ss_rbm
from utils import sharedX, floatX, npy_floatX

class Model():
    def __init__(self):
        rng = numpy.random.RandomState(123)
        self.Wv = sharedX(0.1 * rng.randn(14*14, 10), name='Wv' )
        self.hbias = sharedX(-1 * numpy.ones(10), name='hbias')
        self.alpha = sharedX(0.1 * rng.rand(10), name='alpha')
        self.mu = sharedX(0.1 * numpy.ones(10), name='mu')
        self.lambd = sharedX(1.0 * numpy.ones(10), name='lambd')
        self.bw_s = 1
        self.n_h = 10
        self.input = T.matrix('input')

def test1():
    img_path   = '/data/lisatmp2/desjagui/data/tfd_cn/fold0/train.pkl'
    img_data = pickle.load(open(img_path))
    img0 = img_data.X[0:1]
    img0patch = img0.reshape(48,48)[:14,:14].reshape(1,14*14)

    image_shape = (1, 48, 48)
    patch_shape = (1, 14, 14)
    pool_shape  = (12, 12)
    fakemodel = Model()
    cssrbm = cssrbm_feature_extractor.FeedForwardConvSSRBM(fakemodel,
            image_shape, patch_shape, pool_shape,
            batch_size = 1)

    model = pooled_ss_rbm.PooledSpikeSlabRBM(
            n_h=10, bw_s=1, n_v=14*14,
            lr = {'type':'linear', 'start':1e-3, 'end':1e-3},
            iscales = { 'Wv': 0.01, 'lambd': 0.37, 'mu': .1, 'alpha': 0.37, 'hbias':0.8, },
            truncation_bound = { "v": 4. },
            sp_weight = {'h': 0.}, sp_targ = {'h': 0.1},
            flags = { 'split_norm': 0., 'use_cd': 0., 'use_energy': 0., 'truncated_normal': 0.})
    model.Wv.set_value(fakemodel.Wv.get_value())
    model.hbias.set_value(fakemodel.hbias.get_value())
    model.mu.set_value(fakemodel.mu.get_value())
    model.alpha.set_value(fakemodel.alpha.get_value())
    model.lambd.set_value(fakemodel.lambd.get_value())

    hid = model.h_given_v(model.input)
    f = theano.function([model.input], hid)

    ##### Compare linear operator ####
    numpy.testing.assert_array_almost_equal(
            cssrbm.fromv_func(img0)[:,:,0,0],
            numpy.dot(img0patch, model.Wv.get_value()))

    ##### Compare inference #####
    numpy.testing.assert_array_almost_equal(
        cssrbm.convout_func(img0)[:,:,0,0],
        f(img0patch))

