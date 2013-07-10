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
    def hossrbm_block_g(cls, n_g, n_h, bw_g, bw_h):
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

        return SparsityMask(mask.T, n_g=n_g, n_h=n_h, bw_g=bw_g, bw_h=bw_h)

    @classmethod
    def hossrbm_block_h(cls, n_g, n_h, bw_g, bw_h):
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

        return SparsityMask(mask.T, n_g=n_g, n_h=n_h, bw_g=bw_g, bw_h=bw_h)
 
    @classmethod
    def blockdiag(cls, n_h, n_s, bw_h=1, bw_s=1, dw_s=None):
        """
        Creates a sparsity mask for latent variables pooling over a subset of slab variables.
        :param n_h: number of h-units
        :param n_s: number of slabs
        :param bw: number of slabs variables per h.
        """
        dw_s = bw_s if dw_s is None else dw_s

        # init Wh
        si = 0
        mask = numpy.zeros((n_h, n_s))
        for hi, si in zip(xrange(0, n_h, bw_h), xrange(0, n_s, dw_s)):
            mask[hi:hi+bw_h, si:si+bw_s] = 1.

        return SparsityMask(mask, n_h=n_h, n_s=n_s, bw_h=bw_h, bw_s=bw_s)
        
    """
    Methods to implement a diagonal sparsity_mask
    """
        
    @classmethod
    def diagonal_g(cls, n_g, n_h, width, delta):
        """
        Creates a sparsity mask for g-units, equivalent to an unfactored model.
        :param n_g: number of g-units
        :param n_h: number of h-units
        :param width: diagonal width in the connectivity matrix
        :param delta: how much the diagonal pattern should increase in h for
                      each unit increase in g
        """
        
        # Create the gh connectivity matrix
        gh_conn = numpy.zeros((n_g, n_h))
        si = 1
        for gi in range(n_g):
            for hi in range(n_h):
                h_start = gi * delta - width / 2
                h_end = h_start + width
                if hi >= h_start and hi < h_end:
                    # This gi should be connected with hi via the unit si
                    gh_conn[gi,hi] = si
                    si += 1
        
        # Build the binary mask
        n_s = si - 1
        mask = numpy.zeros((n_s, n_g))
        
        for gi in range(n_g):
            for hi in range(n_h):
                if gh_conn[gi,hi] >= 1:
                    mask[gh_conn[gi,hi]-1,gi] = 1.

        return SparsityMask(mask.T, n_g=n_g, n_h=n_h, width=width, delta=delta)
        
        

    @classmethod
    def diagonal_h(cls, n_g, n_h, width, delta):
        """
        Creates a sparsity mask for g-units, equivalent to an unfactored model.
        :param n_g: number of g-units
        :param n_h: number of h-units
        :param width: diagonal width in the connectivity matrix
        :param delta: how much the diagonal pattern should increase in h for
                      each unit increase in g
        """
        
        # Create the gh connectivity matrix
        gh_conn = numpy.zeros((n_g, n_h))
        si = 1
        for gi in range(n_g):
            for hi in range(n_h):
                h_start = gi * delta - width / 2
                h_end = h_start + width
                if hi >= h_start and hi < h_end:
                    # This gi should be connected with hi via the unit si
                    gh_conn[gi,hi] = si
                    si += 1       
        
        # Build the binary mask
        n_s = si - 1
        mask = numpy.zeros((n_s, n_g))
        
        for gi in range(n_g):
            for hi in range(n_h):
                if gh_conn[gi,hi] >= 1:
                    mask[gh_conn[gi,hi]-1,hi] = 1.

        return SparsityMask(mask.T, n_g=n_g, n_h=n_h, width=width, delta=delta)
