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
parser.add_option('--phi', action='store_true',  dest='phi', default=False)
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
if opts.phi:
    phi = model.phi.get_value().T
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

##############
# PLOT FILTERS
##############

def plot(w):

    nblocks = int(model.n_g / model.bw_g)
    filters_per_block = model.bw_g * model.bw_h

    block_viewer = PatchViewer((model.bw_g, model.bw_h),
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

        for bidx in xrange(nblocks):

            for fidx in xrange(filters_per_block):

                fi = bidx * filters_per_block + fidx
                topo_view = view_converter.design_mat_to_topo_view(w[fi:fi+1,:])

                if opts.color:
                    # display all 3 channels in one color image
                    block_viewer.add_patch(topo_view[0])
                else:
                    # display channels separately
                    block_viewer.add_patch(topo_view[0,:,:,chan_i])

            if opts.splitblocks:
                pl.imshow(block_viewer.image, interpolation=None)
                pl.axis('off')
                pl.title('Wv - block %i, chan %i' % (bidx, chan_i))
                pl.savefig('filters/filters_chan%i_block%i.png' % (bidx, chan_i))

            chan_viewer.add_patch(block_viewer.image - 0.5)
            block_viewer.clear()

        main_viewer.add_patch(chan_viewer.image - 0.5)
        chan_viewer.clear()

    return copy.copy(main_viewer.image)


w_image = plot(wv)
if opts.phi:
    phi_image = plot(phi)

nplots = 2 if opts.phi else 1
viewer = PatchViewer((1,nplots), 
            (w_image.shape[0], 
             w_image.shape[1]),
            is_color = opts.color,
            pad=(20,20))

viewer.add_patch(w_image - 0.5)
if opts.phi:
    viewer.add_patch(phi_image - 0.5)

pl.imshow(viewer.image, interpolation=None)
pl.savefig('filters_%s.png' % opts.path)
pl.close()

if not opts.noshow:
    viewer.show()
