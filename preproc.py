import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from pylearn2.datasets import preprocessing

class PintoLCN(preprocessing.ExamplewisePreprocessor):

    def __init__(self, img_shape, eps=1e-12):
        self.img_shape = img_shape
        self.eps = eps

    def apply(self, dataset, can_fit=True):
        x = dataset.get_design_matrix()

        denseX = T.matrix(dtype=x.dtype)

        image_shape = (len(x),) + self.img_shape
        X = denseX.reshape(image_shape)
        ones_patch = T.ones((1,1,9,9), dtype=x.dtype)

        convout = conv.conv2d(input = X,
                             filters = ones_patch / (9.*9.),
                             image_shape = image_shape,
                             filter_shape = (1, 1, 9, 9),
                             border_mode='full')

        # For each pixel, remove mean of 3x3 neighborhood
        centered_X = X - convout[:,:,4:-4,4:-4]
        
        # Scale down norm of 3x3 patch if norm is bigger than 1
        sum_sqr_XX = conv.conv2d(input = centered_X**2,
                             filters = ones_patch,
                             image_shape = image_shape,
                             filter_shape = (1, 1, 9, 9),
                             border_mode='full')
        denom = T.sqrt(sum_sqr_XX[:,:,4:-4,4:-4])
        xdenom = denom.reshape(X.shape)
        new_X = centered_X / T.largest(1.0, xdenom)
        new_X = T.flatten(new_X, outdim=2)

        f = theano.function([denseX], new_X)
        dataset.set_design_matrix(f(x))


def gaussian_filter_9x9():
    x = numpy.zeros((9,9), dtype='float32')

    def gauss(x, y, sigma=2.0):
        Z = 2 * numpy.pi * sigma**2
        return  1./Z * numpy.exp(-(x**2 + y**2) / (2. * sigma**2))

    for i in xrange(0,9):
        for j in xrange(0,9):
            x[i,j] = gauss(i-4., j-4.)

    return x / numpy.sum(x)


class LeCunLCN(preprocessing.ExamplewisePreprocessor):

    def __init__(self, img_shape, eps=1e-12):
        self.img_shape = img_shape
        self.eps = eps

    def apply(self, dataset, can_fit=True):
        x = dataset.get_design_matrix()

        denseX = T.matrix(dtype=x.dtype)

        image_shape = (len(x),) + self.img_shape
        X = denseX.reshape(image_shape)
        filters = gaussian_filter_9x9().reshape((1,1,9,9))

        convout = conv.conv2d(input = X,
                             filters = filters,
                             image_shape = image_shape,
                             filter_shape = (1, 1, 9, 9),
                             border_mode='full')

        # For each pixel, remove mean of 9x9 neighborhood
        centered_X = X - convout[:,:,4:-4,4:-4]
        
        # Scale down norm of 9x9 patch if norm is bigger than 1
        sum_sqr_XX = conv.conv2d(input = centered_X**2,
                             filters = filters,
                             image_shape = image_shape,
                             filter_shape = (1, 1, 9, 9),
                             border_mode='full')
        denom = T.sqrt(sum_sqr_XX[:,:,4:-4,4:-4])
        per_img_mean = T.mean(T.flatten(denom, outdim=3), axis=2)
        divisor = T.largest(per_img_mean.dimshuffle((0,1,'x','x')), denom)

        new_X = centered_X / divisor
        new_X = T.flatten(new_X, outdim=2)

        f = theano.function([denseX], new_X)
        dataset.set_design_matrix(f(x))
