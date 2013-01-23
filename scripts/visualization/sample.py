import numpy
import pickle
from optparse import OptionParser

import theano
import theano.tensor as T
from pylearn2.utils import serial
from pylearn2.gui.patch_viewer import make_viewer

from DBM import sharedX, floatX, npy_floatX

def softplus(x): return numpy.log(1. + numpy.exp(x))

parser = OptionParser()
parser.add_option('-m', '--model', action='store', type='string', dest='path')
parser.add_option('--width',  action='store', type='int', dest='width')
parser.add_option('--height', action='store', type='int', dest='height')
parser.add_option('--channels',  action='store', type='int', dest='chans', default=1)
parser.add_option('--color', action='store_true',  dest='color')
parser.add_option('--batch_size', action='store',  type='int', dest='batch_size', default=False)
parser.add_option('-n', action='store', dest='n', type='int', default=20)
parser.add_option('--skip', action='store', type='int', dest='skip', default=10)
parser.add_option('--random', action='store_true', dest='random')
parser.add_option('--burnin', action='store', type='int', dest='burnin', default=100)
(opts, args) = parser.parse_args()

# load and recompile model
model = serial.load(opts.path)
#model.batch_size = opts.batch_size
#model.init_samples()
model.do_theano()

###
# Rebuild sampling function to have mean-field values for layer 0
##
neg_updates = model.neg_sampling_updates()
sample_neg_func = theano.function([], [], updates=neg_updates)

if opts.random:
    temp = numpy.random.randint(0,2, size=model.neg_g.get_value().shape)
    model.neg_g.set_value(temp.astype('float32'))
    temp = numpy.random.randint(0,2, size=model.neg_h.get_value().shape)
    model.neg_h.set_value(temp.astype('float32'))
    v_std = numpy.sqrt(1./softplus(model.beta.get_value()))
    temp = numpy.random.normal(0, v_std, size=model.neg_v.get_value().shape)
    model.neg_v.set_value(temp.astype('float32'))

# Burnin of Markov chain.
for i in xrange(opts.burnin):
    sample_neg_func()

# Start actual sampling.
samples = numpy.zeros((opts.batch_size * opts.n, model.n_v))
indices = numpy.arange(0, len(samples), opts.n)

idx = numpy.random.permutation(model.batch_size)[:opts.batch_size]
for t in xrange(opts.n):
    samples[indices,:] = model.neg_ev.get_value()[idx]
    # skip in between plotted samples
    print t
    for i in xrange(opts.skip):
        sample_neg_func()
    indices += 1

img = make_viewer(samples,
                  (opts.batch_size, opts.n),
                  (opts.height, opts.width),
                  is_color=opts.color)
img.show()
