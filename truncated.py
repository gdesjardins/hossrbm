import numpy
import theano
import theano.tensor as T

from theano.printing import Print
from theano.scalar import BinaryScalarOp, upcast_out
from theano.gof import utils
floatX = theano.config.floatX

SQRT2 = numpy.cast[floatX](numpy.sqrt(2))
def tnormal_icdf(size, avg, std, lbound, ubound, theano_rng, dtype):
    """
    Alternative Method:
    sample = -Phi_inv(Phi(-lbound)*(1-u) + Phi(-ubound)*u)
    """

    def Phi(x):
        erfarg = (x - avg) / (std * SQRT2)
        rval = 0.5 * (1. + T.erf(erfarg))
        return rval.astype(dtype)
    
    def Phi_inv(y, eps=3e-8):
        """ eps was calibrated for cublas.erfinv using float32 """
        temp = 2. * y - 1.
        erfinv_input = T.clip(temp, -1+eps, 1-eps)
        rval = avg + std * SQRT2 * T.erfinv(erfinv_input)
        return rval.astype(dtype)

    # center lower and upper bounds based on mean
    u = theano_rng.uniform(size=size, dtype=dtype)

    # Inverse CDF method. When method becomes numerically unstable, we simply
    # return the bounds based on whether avg < lbound, or ubound < avg.
    cdf_range = Phi(ubound) - Phi(lbound)
    sample = T.switch(
                T.or_(
                    T.lt(cdf_range, 3e-8),
                    T.gt(cdf_range, 1-3e-8)),
                T.switch(
                    T.lt(avg, lbound),
                    lbound,
                    ubound),
                Phi_inv(Phi(lbound) + u * cdf_range))

    return sample

truncated_normal = tnormal_icdf


class TruncNormZ(BinaryScalarOp):

    def __init__(self, output_types_preference=None, name=None, compute_log=False):
        super(TruncNormZ, self).__init__(output_types_preference, name)
        self.compute_log = compute_log

    def impl(self, input):
        return input

    def c_code(self, node, name, (a, b), (z,), sub):
        self_compute_log = int(self.compute_log)
        return """
         if (%(self_compute_log)s == 1) {
             double K;
             if (%(a)s > 0) {
                K = %(a)s;
             } else {
                K = %(b)s;
             }
             double temp = _adaptiveSimpsons(_truncSNormPDF, %(a)s, %(b)s, 1e-9, 4, K);
             %(z)s = logf(temp) - 0.5 * K*K;
         }
         else {
             %(z)s = _adaptiveSimpsons(_truncSNormPDF, %(a)s, %(b)s, 1e-9, 4, 0);
         }
         """ % locals()

    def c_support_code(self):
        return (
"""
// For GPU support
#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE
#endif

#ifndef _TRUNCSNORMPDFFUNCDEFINED
#define _TRUNCSNORMPDFFUNCDEFINED
DEVICE double _truncSNormPDF(double x, double K) {
  double Z = sqrtf(2 * M_PI);
  return 1/Z * expf(0.5*K*K - 0.5 * x*x);
}
#endif


/**
 *  Adaptive Simpson's method. Taken from
 *  http://en.wikipedia.org/wiki/Adaptive_Simpson%27s_method
 **/

#ifndef _ADAPTSIMPSONSAUXFUNCDEFINED
#define _ADAPTSIMPSONSAUXFUNCDEFINED
DEVICE double _adaptiveSimpsonsAux(double (*f)(double, double),
                double a, double b, double epsilon,
                double S, double fa, double fb, double fc, int bottom, double K) {
  double c = (a + b)/2, h = b - a;
  double d = (a + c)/2, e = (c + b)/2;
  double fd = f(d, K), fe = f(e, K);
  double Sleft = (h/12)*(fa + 4*fd + fc);
  double Sright = (h/12)*(fc + 4*fe + fb);
  double S2 = Sleft + Sright;
  if (bottom <= 0 || fabs(S2 - S) <= 15*epsilon)
      return S2 + (S2 - S)/15;
  return _adaptiveSimpsonsAux(f, a, c, epsilon/2, Sleft,  fa, fc, fd, bottom-1, K) +
         _adaptiveSimpsonsAux(f, c, b, epsilon/2, Sright, fc, fb, fe, bottom-1, K);
} 
#endif

#ifndef _ADAPTSIMPSONSFUNCDEFINED
#define _ADAPTSIMPSONSFUNCDEFINED
DEVICE double _adaptiveSimpsons(double (*f)(double, double),   // ptr to function
                           double a, double b,  // interval [a,b]
                           double epsilon,  // error tolerance
                           int maxRecursionDepth,
                           double K) {   // recursion cap
  double c = (a + b)/2, h = b - a;
  double fa = f(a, K), fb = f(b, K), fc = f(c, K);
  double S = (h/6)*(fa + 4*fc + fb);
  return _adaptiveSimpsonsAux(f, a, b, epsilon, S, fa, fb, fc, maxRecursionDepth, K);
}
#endif
        """)


    def c_code_cache_version(self):
        return (1,)

    def grad(self, (a, b), (gz, )):
        raise utils.MethodNotDefined("grad", type(self),
                                     self.__class__.__name__)

    def __hash__(self):
        return super(TruncNormZ, self).__hash__() ^ hash(self.compute_log)

    def __eq__(self, other):
        return super(TruncNormZ, self).__eq__(other) and\
               (self.compute_log == other.compute_log)

trunc_norm_z = TruncNormZ(upcast_out, name='trunc_norm_z')


class TruncNormLogZ(TruncNormZ):

    def __init__(self, output_types_preference=None, name=None):
        super(TruncNormLogZ, self).__init__(output_types_preference, name, compute_log=True)

trunc_norm_log_z = TruncNormLogZ(upcast_out, name='trunc_norm_z')

