#!/opt/lisa/os/epd-7.1.2/bin/python
import sys
import copy
import numpy
import pylab as pl
import pickle
import os
from optparse import OptionParser

from theano import function
import theano.tensor as T
import theano

from pylearn2.gui.patch_viewer import make_viewer
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial


parser = OptionParser()
parser.add_option('-m', '--model', action='store', type='string', dest='path')
parser.add_option('--width',  action='store', type='int', dest='width')
parser.add_option('--height', action='store', type='int', dest='height')
parser.add_option('--channels',  action='store', type='int', dest='chans')
parser.add_option('--color', action='store_true',  dest='color', default=False)
parser.add_option('--global',  action='store_false', dest='local',    default=True)
parser.add_option('--preproc', action='store', type='string', dest='preproc')
parser.add_option('--splitblocks', action='store_true', dest='splitblocks')
parser.add_option('--vbias', action='store_true',  dest='add_vbias', default=False)
parser.add_option('--mu', action='store_true',  dest='mu', default=False)
parser.add_option('--noshow', action='store_true',  dest='noshow', default=False)
(opts, args) = parser.parse_args()

nplots = opts.chans
if opts.color:
    assert opts.chans == 3
    nplots = 1

def get_dims(nf):
    num_rows = numpy.floor(numpy.sqrt(nf))
    return (int(num_rows), int(numpy.ceil(nf / num_rows)))

# load model and retrieve parameters
model = serial.load(opts.path)

wv = model.Wv.get_value().T
wvg = model.Wvg.get_value().T
wvh = model.Wvh.get_value().T
if opts.mu:
    wv = wv * model.mu.get_value()[:, None]

# store weight matrix as dataset, in case we have to process them
wv_dataset = DenseDesignMatrix(X=wv)
if opts.preproc:
    fp = open(opts.preproc, 'r')
    preproc = pickle.load(fp)
    fp.close()
    print 'Applying inverse pipeline...'
    preproc.inverse(wv_dataset)
wv = wv_dataset.X

if opts.add_vbias:
    wv += model.vbias.get_value()

########################
# PLOT BILINEAR FILTERS
########################
def plot(w):

    nblocks = int(model.n_g / model.sparse_gmask.bw_g)
    filters_per_block = model.sparse_gmask.bw_g * model.sparse_hmask.bw_h

    block_viewer = PatchViewer((model.sparse_gmask.bw_g, model.sparse_hmask.bw_h),
                               (opts.height, opts.width),
                               is_color = opts.color,
                               pad=(2,2))

    chan_viewer = PatchViewer(get_dims(nblocks),
                              (block_viewer.image.shape[0],
                              block_viewer.image.shape[1]),
                              is_color = opts.color,
                              pad=(5,5))

    main_viewer = PatchViewer(get_dims(nplots),
                              (chan_viewer.image.shape[0],
                               chan_viewer.image.shape[1]),
                              is_color = opts.color,
                              pad=(10,10))

    topo_shape = [opts.height, opts.width, opts.chans]
    view_converter = DefaultViewConverter(topo_shape)

    if opts.splitblocks:
        os.makedirs('filters/')

    for chan_i in xrange(nplots):

        viewer_dims = slice(0, None) if opts.color else chan_i

        for bidx in xrange(nblocks):

            for fidx in xrange(filters_per_block):
                fi = bidx * filters_per_block + fidx
                topo_view = view_converter.design_mat_to_topo_view(w[fi:fi+1,:])
                try:
                    block_viewer.add_patch(topo_view[0,:,:,viewer_dims])
                except:
                    import pdb; pdb.set_trace()

            if opts.splitblocks:
                pl.imshow(block_viewer.image, interpolation='nearest')
                pl.axis('off')
                pl.title('Wv - block %i, chan %i' % (bidx, chan_i))
                pl.savefig('filters/filters_chan%i_block%i.png' % (bidx, chan_i))

            chan_viewer.add_patch(block_viewer.image[:,:,viewer_dims] - 0.5)
            block_viewer.clear()

        main_viewer.add_patch(chan_viewer.image[:,:,viewer_dims] - 0.5)
        chan_viewer.clear()

    return copy.copy(main_viewer.image)


viewer_g = make_viewer(wvg, get_dims(model.n_g), (opts.height, opts.width), is_color=True)
viewer_h = make_viewer(wvh, get_dims(model.n_h), (opts.height, opts.width), is_color=True)
w_image = plot(wv)

viewer = PatchViewer((1, 3),
            (numpy.max((viewer_g.image.shape[0], viewer_h.image.shape[0], w_image.shape[0])), 
             numpy.max((viewer_g.image.shape[1], viewer_h.image.shape[1], w_image.shape[1]))),
            is_color = opts.color,
            pad=(0,10))

viewer_dims = slice(0, None) if opts.color else 0
viewer.add_patch(viewer_g.image[:,:, viewer_dims] - 0.5)
viewer.add_patch(viewer_h.image[:,:, viewer_dims] - 0.5)
viewer.add_patch(w_image[:,:, viewer_dims] - 0.5)

pl.axis('off')
pl.imshow(viewer.image, interpolation='nearest')
pl.savefig('filters_%s.png' % opts.path)
if not opts.noshow:
    pl.show()
