import numpy
from pylearn2.datasets import dense_design_matrix

kernel = numpy.asarray(
            [[1,0,1],
             [0,1,0],
             [1,0,1]])

class NoisyColorToyData(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, center=True, noise_std=0.1, seed=10984721):

        self.rng = numpy.random.RandomState(seed)

        n = 3 # colors
        m = 5 # positions
        width  = len(kernel[0])
        height = len(kernel)
        cols   = (width + 1)*m
        rows   = height

        X = numpy.zeros(( (2**m)*(2**n), rows*cols*n),dtype='float32')
        noisyX = numpy.zeros((1000*X.shape[0], X.shape[1]), dtype='float32')

        idx = 0
        for i in numpy.ndindex(*([2]*m)):
            for j in numpy.ndindex(*([2]*n)):

                example = numpy.zeros((n,rows,cols),dtype='float32')
                template = numpy.zeros((n,rows,width))

                for k in xrange(m):

                    c = k * (width+1)

                    if i[k]:
                        template[0,:,:] = j[0] * kernel
                        template[1,:,:] = j[1] * kernel
                        template[2,:,:] = j[2] * kernel
                        example[:, 0:rows, c:c+width] = template

                X[idx,:] = example.reshape(rows * cols * n)

                idx += 1

        for j in xrange(0, len(noisyX), len(X)):
            noise = noise_std * self.rng.randn(X.shape[0], X.shape[1])
            noisyX[j:j+len(X)] = X + noise

        view_converter = dense_design_matrix.DefaultViewConverter((rows,cols,1))

        if center:
            noisyX -= numpy.mean(noisyX, axis=0)

        super(NoisyColorToyData2,self).__init__(X = noisyX, view_converter = view_converter)

        assert not numpy.any(numpy.isnan(self.X))
