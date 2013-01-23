import theano
import theano.tensor as T

def true_gradient(param, g_k, axis=0):
    """
    Douglas et. al, On Gradient Adaptation With Unit-Norm Constraints.
    See section on "True Gradient Method".
    """
    h_k = g_k -  (g_k*param).sum(axis=0) * param
    theta_k = T.sqrt(1e-8+T.sqr(h_k).sum(axis=0))
    u_k = h_k / theta_k
    return T.cos(theta_k) * param + T.sin(theta_k) * u_k
