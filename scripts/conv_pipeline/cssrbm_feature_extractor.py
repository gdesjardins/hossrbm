import argparse
import numpy
import pickle

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from theano.printing import Print

class FeedForwardConvSSRBM():

    def __init__(self, model, image_shape, patch_shape, pool_shape, batch_size=None):
        """
        :param model: Object of type pooled_ss_rbm.PooledSpikeSlabRBM
        """
        assert model.bw_s == 1
        batch_size = batch_size if batch_size else model.batch_size

        image_shape = (batch_size,) + image_shape
        filter_shape = (model.n_h,) + patch_shape

        input   = model.input.reshape(image_shape)
        filters = model.Wv.T.reshape(filter_shape)
        alpha_prec = T.nnet.softplus(model.alpha)
        alpha_prec = alpha_prec.dimshuffle(('x', 0, 'x', 'x'))
        mu = model.mu.dimshuffle(('x', 0, 'x', 'x'))
        hbias = model.hbias.dimshuffle(('x', 0, 'x', 'x'))

        from_v = conv.conv2d(input = input,
                filters = filters[:,:,::-1,::-1],
                filter_shape = filter_shape,
                image_shape = image_shape)
        self.fromv_func = theano.function([model.input], from_v)
        
        h_mean  = 0.5 * 1./alpha_prec * from_v**2
        h_mean += from_v * mu
        h_mean += hbias
        conv_out = T.nnet.sigmoid(h_mean)
        
        self.convout_func = theano.function([model.input], conv_out)

        maxpool_out = downsample.max_pool_2d(conv_out, pool_shape)

        output_shape = (batch_size, maxpool_out.shape[1:].prod())
        output = maxpool_out.reshape(output_shape)

        self.preproc = theano.function([model.input], output)

def run(model, dataset,
        batch_size=128,
        image_width=48,
        patch_width=14,
        pool_width=12,
        output_width=9,
        output_file='output.pkl'):

    fp = open(model)
    model = pickle.load(fp)
    fp.close()

    fp = open(dataset)
    dataset = pickle.load(fp)
    fp.close()

    image_shape = (1, image_width, image_width)
    patch_shape = (1, patch_width, patch_width)
    pool_shape  = (pool_width, pool_width)
    cssrbm = FeedForwardConvSSRBM(model,
            image_shape, patch_shape, pool_shape,
            batch_size = batch_size)

    newX = numpy.zeros((len(dataset.X), output_width), dtype='float32')
    in_data = numpy.zeros((batch_size, dataset.X.shape[1]), dtype='float32')
    for i in (0, len(dataset.X), batch_size):
        temp = dataset.X[i:i+batch_size]
        in_data[:len(temp)] = temp
        newX[i:i+len(temp)] = cssrbm.preproc(in_data)[:len(temp)]

    dataset.X = newX
    fp = open(output_file, 'w')
    pickle.dump(dataset, fp)
    fp.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path of model .pkl file.')
    parser.add_argument('--dataset', help='Path to data .pkl file.')
    parser.add_argument('--batch_size', type=int, help='Integer, batch size')
    parser.add_argument('--image_width', type=int, help='Width of full-sized training data (assumes square image).')
    parser.add_argument('--patch_width', type=int, help='Width of patch data (assumes square patches).')
    parser.add_argument('--pool_width', type=int, help='Width of max-pooling.')
    parser.add_argument('--output_width', type=int, help='Width of output after max-pooling')
    parser.add_argument('--output_file', help='Name of output file.')
    args = parser.parse_args()

    run(args.model,
            args.dataset,
            args.batch_size,
            args.image_width,
            args.patch_width,
            args.pool_width,
            args.output_width,
            args.output_file)
