import sys
import pylab as pl

import theano
import numpy
import theano.tensor as T

from theano import function
from pylearn2.gui.patch_viewer import make_viewer, PatchViewer
from pylearn2.utils import serial
floatX = theano.config.floatX

def get_dims(nf):
    num_rows = numpy.floor(numpy.sqrt(nf))
    return (int(num_rows), int(numpy.ceil(nf / num_rows)))

if len(sys.argv) != 5:
    print 'Usage:'
    print 'python show_weights <model file> <img width> <img height> <iscolor>'
    sys.exit()

path = sys.argv[1]
imgw = int(sys.argv[2])
imgh = int(sys.argv[3])
is_color = eval(sys.argv[4])
dataset = serial.load("/data/lisatmp/desjagui/data/tfd_cn/data.pkl")
data = dataset.get_design_matrix()

### LOAD MODEL ###
model = serial.load(path)
Wv = model.Wv.get_value()
Wg = model.Wg.get_value().T
Wh = model.Wh.get_value().T

# build theano function to infer mean-field estimates of g and h
vmat = T.matrix()
updates = model.pos_sampling_updates(v=vmat, n_steps=model.pos_sample_steps)
infer_func = function([vmat], [updates['g'], updates['h'], updates['s']])

print "Enter visible index number [0-%i] ('q' to quit'): " % len(data)
temp = raw_input()

gfilt = numpy.zeros((model.n_g, model.n_v))
hfilt = numpy.zeros((model.n_h, model.n_v))

while temp != 'q':
    # setup minibatch
    idx = int(temp)
    x = numpy.tile(data[idx], (model.batch_size,1))

    # now run mean-field for that given input
    results = infer_func(x)
    [qg, qh, qs] = [r[0] for r in results]

    # given y, get the "meta" g-filters
    import pdb; pdb.set_trace()
    g_s = numpy.dot(qh, Wh) * qs
    h_s = numpy.dot(qg, Wg) * qs

    for i in xrange(model.n_g):
        gfilt[i,:] = numpy.dot(Wg[i,:] * g_s, Wv.T)

    for j in xrange(model.n_h):
        hfilt[j,:] = numpy.dot(Wh[j,:] * h_s, Wv.T)

    ########### PLOTTING ###############
    x_viewer  = make_viewer(x, is_color=is_color)
    wg_viewer = make_viewer(gfilt, get_dims(model.n_g), (imgw,imgh), is_color=is_color)
    wh_viewer = make_viewer(hfilt, get_dims(model.n_h), (imgw,imgh), is_color=is_color)
    wv_viewer = make_viewer(Wv.T,  get_dims(model.n_s), (imgw,imgh), is_color=is_color)

    fig = pl.figure()
    pl.subplot(2,2,3); pl.axis('off')
    pl.title('input')
    pl.imshow(x_viewer.image)
    pl.subplot(2,2,1); pl.axis('off')
    pl.title('g-filters')
    pl.imshow(wg_viewer.image)
    pl.subplot(2,2,2); pl.axis('off')
    pl.title('h-filters')
    pl.imshow(wh_viewer.image)
    pl.subplot(2,2,4); pl.axis('off')
    pl.title('Wv filters')
    pl.imshow(wv_viewer.image)
    pl.show()

    print "Enter visible index number [0-%i] ('q' to quit): " % len(data)
    temp = raw_input()
