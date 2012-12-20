import numpy
import theano

def sparsity_mask(type, **kwargs):
    assert hasattr(SparsityMask, type)
    const = getattr(SparsityMask, type)
    return const(**kwargs)

class SparsityMask(object):

    def __init__(self, mask, **kwargs):
        self.mask = numpy.asarray(mask, dtype=theano.config.floatX)
        for (k,v) in kwargs.iteritems():
            setattr(self,k,v)

    @classmethod
    def unfactored_g(cls, n_g, n_h, bw_g, bw_h):
        """
        Creates a sparsity mask for g-units, equivalent to an unfactored model.
        :param n_g: number of g-units
        :param n_h: number of h-units
        :param bw_g: block width in g
        :param bw_h: block width in h
        """
        assert (n_g % bw_g) == 0 and (n_h % bw_h) == 0 and (n_g / bw_g == n_h / bw_h)
        n_s = (n_g / bw_g) * (bw_g * bw_h)

        # init Wg
        si = 0
        mask = numpy.zeros((n_s, n_g))
        for gi in xrange(n_g):
            mask[si:si+bw_h, gi] = 1.
            si += bw_h

        return SparsityMask(mask, n_g=n_g, n_h=n_h, bw_g=bw_g, bw_h=bw_h)

    @classmethod
    def unfactored_h(cls, n_g, n_h, bw_g, bw_h):
        """
        Creates a sparsity mask for h-units, equivalent to an unfactored model.
        :param n_g: number of g-units
        :param n_h: number of h-units
        :param bw_g: block width in g
        :param bw_h: block width in h
        """
        assert (n_g % bw_g) == 0 and (n_h % bw_h) == 0 and (n_g / bw_g == n_h / bw_h)
        n_s = (n_g / bw_g) * (bw_g * bw_h)

        # init Wh
        si = 0
        ds = bw_g * bw_h
        mask = numpy.zeros((n_s, n_h))

        for hi in xrange(n_h):
            bi = hi / bw_h
            mask[bi*ds + hi%bw_h:(bi+1)*ds:bw_h, hi] = 1.

        return SparsityMask(mask, n_g=n_g, n_h=n_h, bw_g=bw_g, bw_h=bw_h)
