"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""
import numpy
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.printing import Print
from hossrbm import BilinearSpikeSlabRBM

floatX = theano.config.floatX

class FactoredBilinearSpikeSlabRBM(BilinearSpikeSlabRBM):

    def params(self):
        """
        Returns a list of learnt model parameters.
        """
        params = [self.scalar_norms, self.Wg, self.Wh, self.Wv]
        params += [self.hbias, self.gbias, self.alpha, self.mu, self.lambd]
        return params

    def h_given_gsv_input(self, g_sample, s_sample, v_sample):
        """
        Compute mean activation of h given (g,s,v)
        :param g_sample: T.matrix of shape (batch_size, n_g matrix)
        :param s_sample: T.matrix of shape (batch_size, n_s matrix)
        :param v_sample: T.matrix of shape (batch_size, n_v matrix)
        """
        from_v = self.from_v(v_sample)
        from_g = self.from_g(g_sample)
        h_mean_s = from_g * s_sample * from_v
        h_mean_s += self.alpha * self.mu * s_sample * from_g
        h_mean_s -= 0.5 * self.alpha * self.mu**2 * from_g
        h_mean = self.to_h(h_mean_s)
        return h_mean
    
    def h_given_gsv(self, g_sample, s_sample, v_sample):
        h_mean = self.h_given_gsv_input(g_sample, s_sample, v_sample)
        return T.nnet.sigmoid(h_mean)

    def sample_h_given_gsv(self, g_sample, s_sample, v_sample, rng=None):
        """
        Generates sample from p(h | g, s, v)
        """
        h_mean = self.h_given_gsv(g_sample, s_sample, v_sample)

        rng = self.theano_rng if rng is None else rng
        h_sample = rng.binomial(size=(self.batch_size,self.n_h),
                                n=1, p=h_mean, dtype=floatX)
        return h_sample

    def g_given_hsv_input(self, h_sample, s_sample, v_sample):
        """
        Compute mean activation of g given (h,s,v).
        :param h_sample: T.matrix of shape (batch_size, n_h matrix)
        :param s_sample: T.matrix of shape (batch_size, n_s matrix)
        :param v_sample: T.matrix of shape (batch_size, n_v matrix)
        """
        from_v = self.from_v(v_sample)
        from_h = self.from_h(h_sample)
        g_mean_s = from_h * s_sample * from_v
        g_mean_s += self.alpha * self.mu * s_sample * from_h
        g_mean_s -= 0.5 * self.alpha * self.mu**2 * from_h
        g_mean = self.to_g(g_mean_s)
        return g_mean
    
    def g_given_hsv(self, h_sample, s_sample, v_sample):
        g_mean = self.g_given_hsv_input(h_sample, s_sample, v_sample)
        return T.nnet.sigmoid(g_mean)

    def sample_g_given_hsv(self, h_sample, s_sample, v_sample, rng=None):
        """
        Generates sample from p(g | h, s, v)
        """
        g_mean = self.g_given_hsv(h_sample, s_sample, v_sample)

        rng = self.theano_rng if rng is None else rng
        g_sample = rng.binomial(size=(self.batch_size,self.n_g),
                                n=1, p=g_mean, dtype=floatX)
        return g_sample


    ##################
    # SAMPLING STUFF #
    ##################
    def pos_sampling(self, v, init_state, n_steps=1):
        """
        Mixed mean-field + sampling inference in positive phase.
        :param v: input being conditioned on
        :param init: dictionary of initial values
        :param n_steps: number of Gibbs updates to perform afterwards.
        """
        def gibbs_iteration(g1, h1, v):
            s2 = self.sample_s_given_ghv(g1, h1, v)
            g2 = self.sample_g_given_hsv(h1, s2, v)
            h2 = self.sample_h_given_gsv(g2, s2, v)
            return [g2, h2, s2]

        [new_g, new_h, new_s], updates = theano.scan(
                gibbs_iteration,
                outputs_info = [init_state['g'], init_state['h'], None],
                non_sequences = [v],
                n_steps=n_steps)

        new_g = new_g[-1]
        new_h = new_h[-1]
        new_s = new_s[-1]

        return [new_g, new_h, new_s]

    def neg_sampling(self, g_sample, h_sample, s_sample, v_sample, n_steps=1):
        """
        Gibbs step for negative phase, which alternates:
        p(g|b,h,v), p(h|b,g,v), p(b|g,h,v), p(s|b,g,h,v) and p(v|b,g,h,s)
        :param g_sample: T.matrix of shape (batch_size, n_g)
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param s_sample: T.matrix of shape (batch_size, n_s)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        :param n_steps: number of Gibbs updates to perform in negative phase.
        """
        def gibbs_iteration(g1, h1, v1):
            s2 = self.sample_s_given_ghv(g1, h1, v1)
            g2 = self.sample_g_given_hsv(h1, s2, v1)
            h2 = self.sample_h_given_gsv(g2, s2, v1)
            v2 = self.sample_v_given_ghs(g2, h2, s2)
            return [g2, h2, s2, v2]

        [new_g, new_h, new_s, new_v] , updates = theano.scan(
                gibbs_iteration,
                outputs_info = [g_sample, h_sample, None, v_sample],
                n_steps=n_steps)

        return [new_g[-1], new_h[-1], new_s[-1], new_v[-1]]

    ####################
    # MEAN-FIELD STUFF #
    ####################

    def mf_g_hat(self, h_hat, s_hat, v):
        return self.g_given_hsv(h_hat, s_hat, v)

    def mf_h_hat(self, g_hat, s_hat, v):
        return self.h_given_gsv(g_hat, s_hat, v)
    
    def mf_s_hat(self, g_hat, h_hat, v):
        return self.s_given_ghv(g_hat, h_hat, v)

    def e_step(self, v, n_steps=100, eps=1e-2):
        new_g = T.ones((v.shape[0], self.n_g)) * T.nnet.sigmoid(self.gbias)
        new_h = T.ones((v.shape[0], self.n_h)) * T.nnet.sigmoid(self.hbias)

        def estep_iteration(g1, h1, v):
            s2 = self.mf_s_hat(g1, h1, v)
            g2 = self.mf_g_hat(h1, s2, v)
            h2 = self.mf_h_hat(g2, s2, v)
            delta = 0.5 * (abs(g2 - g1).mean() + abs(h2 - h1).mean())
            return [g2, h2, s2], theano.scan_module.until(delta < eps)

        [new_g, new_h, new_s], updates = theano.scan(
                    estep_iteration,
                    outputs_info = [new_g, new_h, None],
                    non_sequences = [v],
                    n_steps=n_steps)
        new_g = new_g[-1]
        new_h = new_h[-1]
        new_s = new_s[-1]

        return [new_g, new_h, new_s]

    def e_step_updates(self, v, n_steps=1):
        [new_g, new_h, new_s] = self.e_step(v, n_steps=n_steps)
        updates = OrderedDict()
        updates['g'] = new_g
        updates['h'] = new_h
        updates['s'] = new_s
        return updates


