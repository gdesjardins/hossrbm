"""
This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""
import numpy
import md5
import pickle
import copy
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import function, shared
from theano.sandbox import linalg
from theano.ifelse import ifelse
from theano.sandbox.scan import scan

from pylearn2.training_algorithms import default
from pylearn2.utils import serial
from pylearn2.base import Block
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace

import truncated
import cost as costmod
from utils import tools
from utils import rbm_utils
from utils import sharedX, floatX, npy_floatX
from true_gradient import true_gradient

def sigm(x): return 1./(1 + numpy.exp(-x))
def softplus(x): return numpy.log(1. + numpy.exp(x))
def softplus_inv(x): return numpy.log(numpy.exp(x) - 1.)

def phi(x):
    Z = T.sqrt(2 * npy_floatX(numpy.pi)).astype(floatX)
    return 1/Z * T.exp(-0.5 * x**2)

def Phi(x):
    return 0.5 * (T.erf(x / T.sqrt(2)) + 1)

def trunc_gauss_Z(mu, sigma, a, b):
    alpha = (a - mu) / sigma
    beta  = (b - mu) / sigma
    Z = Phi(beta) - Phi(alpha)
    return Z

def gauss_ent(mu, sigma, a=None, b=None):
    cte = npy_floatX(numpy.sqrt(2 * numpy.pi * numpy.exp(1)))
    if a is None and b is None:
        rval  = T.log(cte * sigma)
    else:
        alpha = (a - mu) / sigma
        beta  = (b - mu) / sigma
        Z = Phi(beta) - Phi(alpha)
        rval  = T.log(cte * sigma * Z)
        rval += (alpha * phi(alpha) - beta * phi(beta)) / (2*Z)
        rval -= (phi(alpha) - phi(beta))**2 / (2*Z**2)
    return rval

class BilinearSpikeSlabRBM(Model, Block):
    """Spike & Slab Restricted Boltzmann Machine (RBM)  """

    def validate_flags(self, flags):
        flags.setdefault('truncate_t', False)
        flags.setdefault('truncate_s', False)
        flags.setdefault('truncate_v', False)
        flags.setdefault('scalar_lambd', False)
        flags.setdefault('wv_true_gradient', False)
        flags.setdefault('wv_norm', None)
        flags.setdefault('ml_lambd', False)
        flags.setdefault('init_mf_rand', False)
        flags.setdefault('center_g', False)
        flags.setdefault('center_h', False)
        flags.setdefault('pos_phase_ch', False)
        flags.setdefault('whiten_s', False)
        flags.setdefault('standardize_s', False)
        flags.setdefault('standardize_t', False)
        if len(flags.keys()) != 14:
            raise NotImplementedError('One or more flags are currently not implemented.')

    def load_params(self, model):
        fp = open(model)
        model = pickle.load(fp)
        fp.close()

        self.lambd.set_value(model.lambd.get_value())
        self.Wv.set_value(model.Wv.get_value())
        self.mu.set_value(model.mu.get_value())
        self.alpha.set_value(model.alpha.get_value())
        self.gamma_s.set_value(model.gamma_s.get_value())
        self.nu.set_value(model.nu.get_value())
        self.beta.set_value(model.beta.get_value())
        self.gamma_t.set_value(model.gamma_t.get_value())
        self.Wh.set_value(model.Wh.get_value())
        self.hbias.set_value(model.hbias.get_value())
        self.Wg.set_value(model.Wg.get_value())
        self.gbias.set_value(model.gbias.get_value())

        # sync negative phase particles
        self.neg_g.set_value(model.neg_g.get_value())
        self.neg_h.set_value(model.neg_h.get_value())
        self.neg_s.set_value(model.neg_s.get_value())
        self.neg_t.set_value(model.neg_t.get_value())
        self.neg_v.set_value(model.neg_v.get_value())

        # sync random number generators
        self.rng.set_state(model.rng.get_state())
        self.theano_rng.rstate = model.theano_rng.rstate
        for (self_rng_state, model_rng_state) in \
                zip(self.theano_rng.state_updates, 
                    model.theano_rng.state_updates):
            self_rng_state[0].set_value(model_rng_state[0].get_value())
        # reset timestamps
        self.batches_seen = model.batches_seen
        self.examples_seen = model.examples_seen
        self.iter.set_value(model.iter.get_value())


    def __init__(self, 
            numpy_rng = None, theano_rng = None,
            n_g=99, n_h=99, n_s=99, n_v=100, init_from=None,
            sparse_gmask = None, sparse_hmask = None,
            pos_steps=1, neg_sample_steps=1,
            lr_spec=None, lr_timestamp=None, lr_mults = {},
            iscales={}, clip_min={}, clip_max={}, truncation_bound={},
            l1 = {}, l2 = {},
            sp_weight={}, sp_targ={},
            batch_size = 13,
            compile=True,
            debug=False,
            seed=1241234,
            my_save_path=None, save_at=None, save_every=None,
            flags = {},
            max_updates = 5e5):
        """
        :param n_h: number of h-hidden units
        :param n_v: number of visible units
        :param iscales: optional dictionary containing initialization scale for each parameter
        :param neg_sample_steps: number of sampling updates to perform in negative phase.
        :param l1: hyper-parameter controlling amount of L1 regularization
        :param l2: hyper-parameter controlling amount of L2 regularization
        :param batch_size: size of positive and negative phase minibatch
        :param compile: compile sampling and learning functions
        :param seed: seed used to initialize numpy and theano RNGs.
        """
        Model.__init__(self)
        Block.__init__(self)
        assert lr_spec is not None
        for k in ['h']: assert k in sp_weight.keys()
        for k in ['h']: assert k in sp_targ.keys()
        self.validate_flags(flags)

        self.jobman_channel = None
        self.jobman_state = {}
        self.register_names_to_del(['jobman_channel'])

        ### make sure all parameters are floatX ###
        for (k,v) in l1.iteritems(): l1[k] = npy_floatX(v)
        for (k,v) in l2.iteritems(): l2[k] = npy_floatX(v)
        for (k,v) in sp_weight.iteritems(): sp_weight[k] = npy_floatX(v)
        for (k,v) in sp_targ.iteritems(): sp_targ[k] = npy_floatX(v)
        for (k,v) in clip_min.iteritems(): clip_min[k] = npy_floatX(v)
        for (k,v) in clip_max.iteritems(): clip_max[k] = npy_floatX(v)

        # dump initialization parameters to object
        for (k,v) in locals().iteritems():
            if k!='self': setattr(self,k,v)

        # allocate random number generators
        self.rng = numpy.random.RandomState(seed) if numpy_rng is None else numpy_rng
        self.theano_rng = RandomStreams(self.rng.randint(2**30)) if theano_rng is None else theano_rng

        ############### ALLOCATE PARAMETERS #################
        # allocate symbolic variable for input
        self.input = T.matrix('input')
        self.init_parameters()
        self.init_chains()

        # learning rate, with deferred 1./t annealing
        self.iter = sharedX(0.0, name='iter')

        if lr_spec['type'] == 'anneal':
            num = lr_spec['init'] * lr_spec['start'] 
            denum = T.maximum(lr_spec['start'], lr_spec['slope'] * self.iter)
            self.lr = T.maximum(lr_spec['floor'], num/denum) 
        elif lr_spec['type'] == 'linear':
            lr_start = npy_floatX(lr_spec['start'])
            lr_end   = npy_floatX(lr_spec['end'])
            self.lr = lr_start + self.iter * (lr_end - lr_start) / npy_floatX(self.max_updates)
        else:
            raise ValueError('Incorrect value for lr_spec[type]')

        # configure input-space (new pylearn2 feature?)
        self.input_space = VectorSpace(n_v)
        self.output_space = VectorSpace(n_h)

        self.batches_seen = 0                    # incremented on every batch
        self.examples_seen = 0                   # incremented on every training example
        self.force_batch_size = batch_size  # force minibatch size

        self.error_record = []
 
        if compile: self.do_theano()

        #### load layer 1 parameters from file ####
        if init_from:
            self.load_params(init_from)

    def init_weight(self, iscale, shape, name, normalize=False, axis=0):
        value = self.rng.normal(size=shape) * iscale
        if normalize:
            value /= numpy.sqrt(numpy.sum(value**2, axis=axis))
        return sharedX(value, name=name)

    def init_parameters(self):
        assert self.sparse_hmask

        self.linproj_v_std = sharedX(numpy.ones(self.n_s), 'linproj_v_std')
        self.linproj_s_std = sharedX(numpy.ones(self.n_g), 'linproj_s_std')
        self.gamma_s = sharedX(numpy.ones(self.n_s), 'gamma_s')
        self.gamma_t = sharedX(numpy.ones(self.n_g), 'gamma_t')

        # Init (visible, slabs) weight matrix.
        self.Wv = self.init_weight(self.iscales['Wv'], (self.n_v, self.n_s), 'Wv',
                normalize = self.flags['wv_norm'] == 'unit')
        self._Wv = self.Wv / self.gamma_s
        self.norm_wv = T.sqrt(T.sum(self.Wv**2, axis=0))

        # mean parameter on s
        self.mu = sharedX(self.iscales['mu'] * numpy.ones(self.n_s), name='mu')
        self._mu = self.mu * self.gamma_s
        # precision (alpha) parameters on s
        self.alpha = sharedX(self.iscales['alpha'] * numpy.ones(self.n_s), name='alpha')
        self.alpha_prec = T.nnet.softplus(self.alpha)
        self.alpha_sigma = T.sqrt(1./self.alpha_prec)

        # mean parameter on t (no pooling)
        self.nu = sharedX(self.iscales['nu'] * numpy.ones(self.n_g), name='nu')
        self._nu = self.nu * self.gamma_t
        # precision parameters on t
        self.beta = sharedX(self.iscales['beta'] * numpy.ones(self.n_g), name='beta')
        self.beta_prec = T.nnet.softplus(self.beta)
        self.beta_sigma = T.sqrt(1./self.beta_prec)

        # Initialize (slab, hidden) pooling matrix
        self.Wh = sharedX(self.sparse_hmask.mask.T * self.iscales.get('Wh', 1.0), name='Wh')

        if self.sparse_gmask:
            self.Wg = sharedX(self.sparse_gmask.mask.T * self.iscales.get('Wg', 1.0), name='Wg')
        else:
            self.Wg = self.init_weight(self.iscales['Wg'], (self.n_s, self.n_g), 'Wg')
        self._Wg = self.Wg / self.gamma_t

        # allocate shared variables for bias parameters
        self.gbias = sharedX(self.iscales['gbias'] * numpy.ones(self.n_g), name='gbias')
        self.hbias = sharedX(self.iscales['hbias'] * numpy.ones(self.n_h), name='hbias')
        self.cg = sharedX(0. * numpy.ones(self.n_g), name='cg')
        self.ch = sharedX(0. * numpy.ones(self.n_h), name='ch')

        # diagonal of precision matrix of visible units
        self.lambd = sharedX(self.iscales['lambd'] * numpy.ones(self.n_v), name='lambd')
        self.lambd_prec = T.nnet.softplus(self.lambd)

    def init_chains(self):
        """ Allocate shared variable for persistent chain """
        # initialize buffers to store inference state
        self.pos_g  = sharedX(numpy.zeros((self.batch_size, self.n_g)), name='pos_g')
        self.pos_t1 = sharedX(numpy.zeros((self.batch_size, self.n_g)), name='pos_t1')
        self.pos_h  = sharedX(numpy.zeros((self.batch_size, self.n_h)), name='pos_h')
        self.pos_s1 = sharedX(numpy.zeros((self.batch_size, self.n_s)), name='pos_s1')
        self.pos_s1_var = sharedX(numpy.zeros((self.batch_size, self.n_s)), name='pos_s1_var')
        self.pos_s0 = sharedX(numpy.zeros((self.batch_size, self.n_s)), name='pos_s0')
        self.pos_s0_var = sharedX(numpy.zeros((self.batch_size, self.n_s)), name='pos_s0_var')
        # initialize visible unit chains
        scale = numpy.sqrt(1./softplus(self.lambd.get_value()))
        neg_v  = self.rng.normal(loc=0, scale=scale, size=(self.batch_size, self.n_v))
        self.neg_v  = sharedX(neg_v, name='neg_v')
        # initialize t-chain
        scale = numpy.sqrt(1./softplus(self.beta.get_value()))
        neg_t  = self.rng.normal(loc=0., scale=scale, size=(self.batch_size, self.n_g))
        self.neg_t  = sharedX(neg_t, name='neg_t')
        # initialize s-chain
        scale = numpy.sqrt(1./softplus(self.alpha.get_value()))
        neg_s  = self.rng.normal(loc=0., scale=scale, size=(self.batch_size, self.n_s))
        self.neg_s  = sharedX(neg_s, name='neg_s')
        # initialize binary g-h chains
        pval_g = sigm(self.gbias.get_value())
        pval_h = sigm(self.hbias.get_value())
        neg_g = self.rng.binomial(n=1, p=pval_g, size=(self.batch_size, self.n_g))
        neg_h = self.rng.binomial(n=1, p=pval_h, size=(self.batch_size, self.n_h))
        self.neg_h  = sharedX(neg_h, name='neg_h')
        self.neg_g  = sharedX(neg_g, name='neg_g')
        # other misc.
        self.pos_counter = sharedX(0., name='pos_counter')
        self.odd_even = sharedX(0., name='odd_even')
 
    def params(self):
        """
        Returns a list of learnt model parameters.
        """
        params = [self.Wv, self.Wg, self.hbias, self.gbias, self.mu, self.alpha, self.nu, self.beta, self.lambd]
        return params

    def do_theano(self):
        """ Compiles all theano functions needed to use the model"""

        init_names = dir(self)

        ###### All fields you don't want to get pickled (e.g., theano functions) should be created below this line
        self.init_debug()

        # SAMPLING: NEGATIVE PHASE
        neg_updates = self.neg_sampling_updates(n_steps=self.neg_sample_steps)
        self.sample_func = theano.function([], [], updates=neg_updates)

        # POSITIVE PHASE
        pos_updates = self.pos_phase_updates(
                self.input,
                n_steps = self.pos_steps)

        self.inference_func = theano.function([self.input], [],
                updates=pos_updates)

        ##
        # BUILD COST OBJECTS
        ##
        lcost = self.ml_cost(
                        pos_g = self.pos_g,
                        pos_t1 = self.pos_t1,
                        pos_h = self.pos_h,
                        pos_s1 = self.pos_s1,
                        pos_s1_var = self.pos_s1_var,
                        pos_s0 = self.pos_s0,
                        pos_s0_var = self.pos_s0_var,
                        pos_v = self.input,
                        neg_g = neg_updates[self.neg_g],
                        neg_t = neg_updates[self.neg_t],
                        neg_h = neg_updates[self.neg_h],
                        neg_s = neg_updates[self.neg_s],
                        neg_v = neg_updates[self.neg_v])

        #spcost = self.get_sparsity_cost(
                        #pos_g = pos_updates[self.pos_g],
                        #pos_h = pos_updates[self.pos_h])

        regcost = self.get_reg_cost(self.l2, self.l1)

        ##
        # COMPUTE GRADIENTS WRT. COSTS
        ##
        #main_cost = [lcost, spcost, regcost]
        main_cost = [lcost, regcost]

        learning_grads = costmod.compute_gradients(self.lr, self.lr_mults, *main_cost)

        weight_updates = OrderedDict()
        if self.flags['wv_true_gradient']:
            weight_updates[self.Wv] = true_gradient(self.Wv, -learning_grads[self.Wv])

        ##
        # BUILD UPDATES DICTIONARY FROM GRADIENTS
        ##
        learning_updates = costmod.get_updates(learning_grads)
        learning_updates.update(neg_updates)
        learning_updates.update({self.iter: self.iter+1})
        learning_updates.update(weight_updates)
        learning_updates.update(self.statistics_updates(self.input, self.pos_h, self.pos_s1, self.pos_s0))

        # build theano function to train on a single minibatch
        self.batch_train_func = function([self.input], [],
                                         updates=learning_updates,
                                         name='train_rbm_func')
        #theano.printing.pydotprint(self.batch_train_func, outfile='batch_train_func.png', scan_graphs=True);

        #######################
        # CONSTRAINT FUNCTION #
        #######################
        constraint_updates = self.get_constraint_updates()
        self.enforce_constraints = theano.function([],[], updates=constraint_updates)

        ###### All fields you don't want to get pickled should be created above this line
        final_names = dir(self)
        self.register_names_to_del( [ name for name in (final_names) if name not in init_names ])

        # Before we start learning, make sure constraints are enforced
        self.enforce_constraints()

    def statistics_updates(self, v_sample, h_sample, s1, s0):
        from_h = self.from_h(h_sample, center=False)
        pos_s = from_h * s1 + (1-from_h) * s0
        updates = OrderedDict()
        # monitor statistics of linear projection of v
        linproj_v = 1./self.alpha_prec * self.from_v(v_sample)
        updates[self.linproj_v_std] = linproj_v.std(axis=0)
        # monitor statistics of linear projection of s
        linproj_s = 1./self.beta_prec * self.from_s(pos_s)
        updates[self.linproj_s_std] = linproj_s.std(axis=0)
        return updates

    def get_constraint_updates(self):
        updates = OrderedDict() 

        if self.flags['wv_norm'] == 'unit':
            updates[self.Wv] = self.Wv / self.norm_wv
        elif self.flags['wv_norm'] == 'max_unit':
            updates[self.Wv] = self.Wv / self.norm_wv * T.minimum(self.norm_wv, 1.0)

        if self.flags['scalar_lambd']:
            updates[self.lambd] = T.mean(self.lambd) * T.ones_like(self.lambd)

        if self.flags['standardize_s']:
            new_gamma_s = 0.999 * self.gamma_s + 0.001 * self.linproj_v_std
            updates[self.gamma_s] = new_gamma_s
            updates[self.Wv] = new_gamma_s / self.gamma_s * self.Wv
            updates[self.mu] = self.gamma_s / new_gamma_s * self.mu

        if self.flags['standardize_t']:
            new_gamma_t = 0.999 * self.gamma_t + 0.001 * self.linproj_s_std
            updates[self.gamma_t] = new_gamma_t
            updates[self.Wg] = new_gamma_t / self.gamma_t * self.Wg
            updates[self.nu] = self.gamma_t / new_gamma_t * self.nu

        ## Enforce sparsity pattern on g if required ##
        if self.sparse_gmask:
            updates[self.Wg] = self.Wg * self.sparse_gmask.mask.T

        ## clip parameters to maximum values (if applicable)
        for (k,v) in self.clip_max.iteritems():
            assert k in [param.name for param in self.params()]
            param = updates.get(k, getattr(self, k))
            updates[param] = T.clip(param, param, v)

        ## clip parameters to minimum values (if applicable)
        for (k,v) in self.clip_min.iteritems():
            assert k in [param.name for param in self.params()]
            param = updates.get(k, getattr(self, k))
            updates[param] = T.clip(updates.get(param, param), v, param)

        return updates

    def train_batch(self, dataset, batch_size):

        x = dataset.get_batch_design(batch_size, include_labels=False)
        if self.flags['truncate_v']:
            x = numpy.clip(x, -self.truncation_bound['v'], self.truncation_bound['v'])

        self.inference_func(x)
        self.batch_train_func(x)
        self.enforce_constraints()

        # accounting...
        self.examples_seen += self.batch_size
        self.batches_seen += 1

        # save to different path each epoch
        if self.my_save_path and \
           (self.batches_seen in self.save_at or
            self.batches_seen % self.save_every == 0):
            fname = self.my_save_path + '_e%i.pkl' % self.batches_seen
            print 'Saving to %s ...' % fname,
            serial.save(fname, self)
            print 'done'

        return self.batches_seen < self.max_updates

    def energy(self, g_sample, t_sample, h_sample, s_sample, v_sample):
        from_h = self.from_h(h_sample)
        from_s = self.from_s(s_sample)
        from_v = self.from_v(v_sample)
        cg_sample = g_sample - self.cg if self.flags['center_g'] else g_sample
        ch_sample = h_sample - self.ch if self.flags['center_h'] else h_sample

        energy  = 0.
        energy += T.sum(0.5 * self.lambd_prec * v_sample**2, axis=1)
        energy -= T.sum(from_v * (self._mu + s_sample) * from_h, axis=1)
        energy += 0.5 * T.sum(self.alpha_prec * s_sample**2, axis=1)
        energy -= T.sum(from_s * (self._nu + t_sample) * cg_sample, axis=1)
        energy += 0.5 * T.sum(self.beta_prec * t_sample**2, axis=1)
        energy -= T.dot(cg_sample, self.gbias)
        energy -= T.dot(ch_sample, self.hbias)
        return energy, [g_sample, t_sample, h_sample, s_sample, v_sample]

    def lbound_plus(self, g, t1_mean, h, s1_mean, s1_var, s0_mean, s0_var, v,
                    s_stats_const=True, t_stats_const=True):
        """
        Computes the expectation (under the variational distribution q(g,h)=q(g)q(h)) of the
        log un-normalized probability, i.e. log p^*(g,h,s,v)
        """
        # center variables
        from_h = self.from_h(h, center=False)
        cg = g - self.cg if self.flags['center_g'] else g
        ch = h - self.ch if self.flags['center_h'] else h

        # E[s] = E[s|h=1] p(h=1) + E[s|h=0] p(h=0)
        s = from_h * s1_mean + (1-from_h) * s0_mean
        t = g * t1_mean + (1-g) * 0.
        # E[s^2] = (E[s|h=1]^2 + 1./alpha) p(h=1) + (E[s|h=0]^2 + 1./alpha) p(h=0)
        ss = from_h * (s1_mean**2 + s1_var) + (1-from_h) * (s0_mean**2 + s0_var)
        tt = g * (t1_mean**2 + 1./self.beta_prec)  + (1-g) * (0 + 1./self.beta_prec)
        
        from_v = self.from_v(v)
        from_s = self.from_s(s)

        lq  = 0.
        lq -= 0.5 * T.sum(self.lambd_prec * v**2, axis=1)
        lq -= 0.5 * T.sum(self.alpha_prec * ss, axis=1)
        lq -= 0.5 * T.sum(self.beta_prec * tt, axis=1)

        lq += T.sum(from_v * (self._mu + s1_mean) * from_h, axis=1)
        if self.flags['center_h']:
            lq -= T.sum(from_v * (self._mu + s) * T.dot(self.ch, self.Wh.T), axis=1)

        lq += T.sum(from_s * (self._nu + t1_mean) * g, axis=1)
        if self.flags['center_g']:
            lq -= T.sum(from_s * (self._nu + t) * self.cg, axis=1)

        lq += T.dot(cg, self.gbias)
        lq += T.dot(ch, self.hbias)

        cte = [g, t1_mean, h, s1_mean, s1_var, s0_mean, s0_var, v]
        if s_stats_const:
            cte += [s, ss]
        if t_stats_const:
            cte += [t, tt]

        return lq, cte

    def entropy_term(self, g, t1_mu, h, s1_mu, s0_mu):
        b = self.truncation_bound['s'] if self.flags['truncate_s'] else None
        a = -b if self.flags['truncate_s'] else None
        # entropy from q(h,s)
        rval  = T.sum(-T.xlogx.xlogx(h) - T.xlogx.xlogx(1-h), axis=1)
        rval += T.sum(h * gauss_ent(s1_mu, self.alpha_sigma, a, b), axis=1)
        rval += T.sum((1-h) * gauss_ent(s0_mu, self.alpha_sigma, a, b), axis=1)
        # entropy from q(g,t)
        rval += T.sum(-T.xlogx.xlogx(g) - T.xlogx.xlogx(1-g), axis=1)
        cte = [g, t1_mu, h, s1_mu, s0_mu]
        return rval, cte

    def lbound(self, g, t1_mu, t1_mean, h, s1_mu, s1_mean, s1_var,
               s0_mu, s0_mean, s0_var, v,
               s_stats_const=True, t_stats_const=True):

        rval1, cte1 = self.lbound_plus(g, t1_mean, h,
                s1_mean, s1_var, s0_mean, s0_var, v,
                s_stats_const, t_stats_const)
        rval2, cte2 = self.entropy_term(g, t1_mu, h, s1_mu, s0_mu)
        return rval1 + rval2, cte1 + cte2

    def __call__(self, v, output_type='g+h'):
        print 'Building representation with %s' % output_type
        init_state = OrderedDict()
        init_state['g'] = T.ones((v.shape[0],self.n_g)) * T.nnet.sigmoid(self.gbias)
        init_state['h'] = T.ones((v.shape[0],self.n_h)) * T.nnet.sigmoid(self.hbias)
        [g, h, s2_1, s2_0, v, pos_counter] = self.pos_phase(v, init_state, n_steps=self.pos_steps)
        s = self.s_hat(h, s2_1, s2_0)

        atoms = {
            'g_s' : self.from_g(g),  # g in s-space
            'h_s' : self.from_h(h),  # h in s-space
            's_g' : T.sqrt(self.to_g(s**2)),
            's_h' : T.sqrt(self.to_h(s**2)),
            's_g__h' : T.sqrt(self.to_g(s**2 * self.from_h(h))),
            's_h__g' : T.sqrt(self.to_h(s**2 * self.from_g(g))),
        }

        output_prods = {
            ## factored representations
            'g' : g,
            'h' : h,
            's' : s,
            'gh' : (g.dimshuffle(0,1,'x') * h.dimshuffle(0,'x',1)).flatten(ndim=2),
            'gs': g * atoms['s_g'],
            'hs': h * atoms['s_h'],
            's_g': atoms['s_g'],
            's_h': atoms['s_h'],
            ## unfactored representations
            'sg_s' : atoms['g_s'] * s,
            'sh_s' : atoms['h_s'] * s,
        }

        toks = output_type.split('+')
        output = output_prods[toks[0]]
        for tok in toks[1:]:
            output = T.horizontal_stack(output, output_prods[tok])

        return output

    ######################################
    # MATH FOR CONDITIONAL DISTRIBUTIONS #
    ######################################

    def from_v(self, v_sample):
        return T.dot(self.lambd_prec * v_sample, self._Wv)
    
    def from_s(self, s_sample):
        return T.dot(self.alpha_prec * s_sample, self._Wg)

    def from_gt(self, g_sample, t_sample):
        if self.flags['center_g']:
            g_sample = g_sample - self.cg
        return T.dot((self._nu + t_sample) * g_sample, self._Wg.T)

    def from_h(self, h_sample, center=True):
        if center and self.flags['center_h']:
            h_sample = h_sample - self.ch
        return T.dot(h_sample, self.Wh.T)

    def to_h(self, h_s):
        return T.dot(h_s, self.Wh)

    def g_given_s(self, s_sample):
        from_s = self.from_s(s_sample) 
        g_mean = from_s * self._nu
        g_mean += 0.5 * 1./self.beta_prec * from_s**2
        g_mean += self.gbias
        return T.nnet.sigmoid(g_mean)

    def sample_g_given_s(self, s_sample, rng=None, size=None):
        """
        Generates sample from p(g | s)
        """
        g_mean = self.g_given_s(s_sample)

        rng = self.theano_rng if rng is None else rng
        size = size if size else self.batch_size
        g_sample = rng.binomial(size=(size, self.n_g),
                                n=1, p=g_mean, dtype=floatX)
        return g_sample

    def t_given_gs(self, g_sample, s_sample):
        if self.flags['center_g']:
            g_sample = g_sample - self.cg
        from_s = self.from_s(s_sample)
        t_mean = 1./self.beta_prec * from_s * g_sample
        return t_mean

    def sample_t_given_gs(self, g_sample, s_sample, rng=None, size=None):
        """
        Generates sample from p(t | g, s)
        """
        t_mean = self.t_given_gs(g_sample, s_sample)
        
        rng = self.theano_rng if rng is None else rng
        size = size if size else self.batch_size
        if self.flags['truncate_t']:
            t_sample = truncated.truncated_normal(
                    size=(size, self.n_g),
                    avg = t_mean, 
                    std = T.sqrt(1./self.beta_prec),
                    lbound = -self.truncation_bound['t'],
                    ubound = self.truncation_bound['t'],
                    theano_rng = rng,
                    dtype=floatX)
        else:
            t_sample = rng.normal(
                    size=(size, self.n_g),
                    avg = t_mean, 
                    std = T.sqrt(1./self.beta_prec),
                    dtype=floatX)
        return t_sample

    def h_given_gtv(self, g_sample, t_sample, v_sample):
        from_v = self.from_v(v_sample)
        from_gt = self.from_gt(g_sample, t_sample)
        h_mean_s = from_v * (self._mu + from_gt)

        squared_term = 0.5 * 1./self.alpha_prec * from_v**2
        if self.flags['center_h']:
            squared_term *= (1 - 2*self.ch)
        h_mean_s += squared_term

        if self.flags['truncate_s']:
            b = self.truncation_bound['s']

            h_ones  = T.ones((self.batch_size, self.n_h))
            h_zeros = T.zeros((self.batch_size, self.n_h))
            if self.flags['center_h']:
                h_ones -= self.ch
                h_zeros-= self.ch

            s_mu_h1 = self.s_mu_given_gthv(g_sample, t_sample, h_ones, v_sample)
            s_mu_h0 = self.s_mu_given_gthv(g_sample, t_sample, h_zeros, v_sample)

            Z_s1 = trunc_gauss_Z(s_mu_h1, self.alpha_sigma, -b, b)
            Z_s0 = trunc_gauss_Z(s_mu_h0, self.alpha_sigma, -b, b)
            h_mean_s += T.log(Z_s1) - T.log(Z_s0)

        h_mean = self.to_h(h_mean_s) + self.hbias

        return T.nnet.sigmoid(h_mean)
    
    def sample_h_given_gtv(self, g_sample, t_sample, v_sample, rng=None, size=None):
        """
        Generates sample from p(h | g, t, v)
        """
        h_mean = self.h_given_gtv(g_sample, t_sample, v_sample)

        rng = self.theano_rng if rng is None else rng
        size = size if size else self.batch_size
        h_sample = rng.binomial(size=(size, self.n_h),
                                n=1, p=h_mean, dtype=floatX)
        return h_sample

    def s_mu_given_gthv(self, g_sample, t_sample, h_sample, v_sample):
        from_gt = self.from_gt(g_sample, t_sample)
        from_h = self.from_h(h_sample)
        from_v = self.from_v(v_sample)
        s_mu = 1./self.alpha_prec * from_v * from_h + from_gt
        return s_mu

    def s_stats(self, s_mu):
        """ WARNING: mean of a truncated gaussian is not the mu parameter ! """
        s_mean = s_mu
        s_var = T.tile(T.shape_padleft(1./self.alpha_prec), (self.batch_size, 1))

        if self.flags['truncate_s']:
            alpha = (-self.truncation_bound['s'] - s_mu) / self.alpha_sigma
            beta  = (+self.truncation_bound['s'] - s_mu) / self.alpha_sigma
            Z = Phi(beta) - Phi(alpha)
            s_mean += self.alpha_sigma * (phi(alpha) - phi(beta)) / Z

            s_var_term1 = (alpha * phi(alpha) - beta * phi(beta)) / Z
            s_var_term2 = (phi(alpha) - phi(beta) / Z )**2
            s_var *= (1 + s_var_term1 - s_var_term2)

        return (s_mean, s_var)

    def sample_s_given_gthv(self, g_sample, t_sample, h_sample, v_sample, rng=None, size=None):
        """
        Generates sample from p(s | g, t, h, v)
        """
        s_mu = self.s_mu_given_gthv(g_sample, t_sample, h_sample, v_sample)
        
        rng = self.theano_rng if rng is None else rng
        size = size if size else self.batch_size
        if self.flags['truncate_s']:
            s_sample = truncated.truncated_normal(
                    size=(size, self.n_s),
                    avg = s_mu, 
                    std = T.sqrt(1./self.alpha_prec),
                    lbound = -self.truncation_bound['s'],
                    ubound = +self.truncation_bound['s'],
                    theano_rng = rng,
                    dtype=floatX)
        else:
            s_sample = rng.normal(
                    size=(size, self.n_s),
                    avg = s_mu, 
                    std = T.sqrt(1./self.alpha_prec),
                    dtype=floatX)
        return s_sample

    def v_given_hs(self, h_sample, s_sample):
        from_h = self.from_h(h_sample)
        v_mean =  T.dot(from_h * (self._mu + s_sample), self._Wv.T)
        return v_mean

    def sample_v_given_hs(self, h_sample, s_sample, rng=None, size=None):
        """
        Generates sample from p(v | h, s)
        """
        v_mean = self.v_given_hs(h_sample, s_sample)

        rng = self.theano_rng if rng is None else rng
        size = size if size else self.batch_size
        if self.flags['truncate_v']:
            v_sample = truncated.truncated_normal(
                    size=(size, self.n_v),
                    avg = v_mean, 
                    std = T.sqrt(1./self.lambd_prec),
                    lbound = -self.truncation_bound['v'],
                    ubound = self.truncation_bound['v'],
                    theano_rng = rng,
                    dtype=floatX)
        else:
            v_sample = rng.normal(
                    size=(size, self.n_v),
                    avg = v_mean, 
                    std = T.sqrt(1./self.lambd_prec),
                    dtype=floatX)

        return v_sample

    ##################
    # SAMPLING STUFF #
    ##################

    def pos_phase(self, v, init_state, n_steps=1, eps=1e-3):
        """
        Mixed mean-field + sampling inference in positive phase.
        :param v: input being conditioned on
        :param init: dictionary of initial values
        :param n_steps: number of Gibbs updates to perform afterwards.
        """
        def pos_mf_iteration(g, t1, h, v, pos_counter):

            # new_h := E[h=1]
            new_h = self.h_given_gtv(g, t1, v)

            # new_s := E[s|h=1] p(h=1) + E[s|h=0] p(h=0)
            new_s1_mu = self.s_mu_given_gthv(g, t1, T.ones((v.shape[0],  self.n_h)), v)
            new_s0_mu = self.s_mu_given_gthv(g, t1, T.zeros((v.shape[0], self.n_h)), v)

            # IMPORTANT: for truncation gaussian, E[s1] != mu 
            (new_s1_mean, new_s1_var) = self.s_stats(new_s1_mu)
            (new_s0_mean, new_s0_var) = self.s_stats(new_s0_mu)
            new_s = new_s1_mean * self.from_h(new_h, center=False) +\
                    new_s0_mean * (1-self.from_h(new_h, center=False))

            # new_g := E[g=1] 
            new_g = self.g_given_s(new_s)

            # new_tX := E[t|g=X] 
            new_t1 = self.t_given_gs(T.ones((v.shape[0], self.n_g)), new_s)

            # stopping criterion
            dlbound_args = [new_g, new_t1, new_h,
                            new_s1_mu, new_s1_mean, new_s1_var,
                            new_s0_mu, new_s0_mean, new_s0_var, v]

            dl_dghat = T.max(abs(self.dlbound_dg(*dlbound_args)))
            dl_dhhat = T.max(abs(self.dlbound_dh(*dlbound_args)))
            stop = T.maximum(dl_dghat, dl_dhhat)

            return [new_g, new_t1, new_h,
                    new_s1_mean, new_s1_var,
                    new_s0_mean, new_s0_var, v,
                    pos_counter+1],\
                   theano.scan_module.until(stop < eps)

        states = [T.unbroadcast(T.shape_padleft(init_state['g'])),
                  T.unbroadcast(T.shape_padleft(init_state['t'])),
                  T.unbroadcast(T.shape_padleft(init_state['h'])),
                  {'steps': 1}, {'steps': 1},
                  {'steps': 1}, {'steps': 1},
                  T.unbroadcast(T.shape_padleft(v)),
                  T.unbroadcast(T.shape_padleft(0.))
                  ]

        rvals, updates = scan(
                pos_mf_iteration,
                states = states,
                n_steps=n_steps)

        return [rval[0] for rval in rvals]

    def pos_phase_updates(self, v, init_state=None, n_steps=1):
        """
        Implements the positive phase sampling, which performs blocks Gibbs
        sampling in order to sample from p(g,h,x,y|v).
        :param v: fixed training set
        :param init: dictionary of initial values, or None if sampling from scratch
        :param n_steps: scalar, number of Gibbs steps to perform.
        :param restart: if False, start sampling from buffers self.pos_*
        """
        if init_state is None:
            assert n_steps
            # start sampler from scratch
            init_state = OrderedDict()
            init_state['g'] = T.ones((v.shape[0], self.n_g)) * T.nnet.sigmoid(self.gbias)
            init_state['t'] = T.zeros((v.shape[0], self.n_g))
            init_state['h'] = T.ones((v.shape[0], self.n_h)) * T.nnet.sigmoid(self.hbias)

        rval = self.pos_phase(v, init_state=init_state, n_steps=n_steps)

        # update running average of positive phase activations
        pos_updates = OrderedDict()
        pos_updates[self.odd_even] = (self.odd_even + 1) % 2
        pos_updates[self.pos_g]  = rval[0]
        pos_updates[self.pos_t1] = rval[1]
        pos_updates[self.pos_h]  = rval[2]
        pos_updates[self.pos_s1] = rval[3]
        pos_updates[self.pos_s1_var] = rval[4]
        pos_updates[self.pos_s0] = rval[5]
        pos_updates[self.pos_s0_var] = rval[6]
        # rval[7] is not used.
        pos_updates[self.pos_counter] = rval[8]

        if self.flags['pos_phase_ch']:
            # TODO: move to statistics_updates
            pos_h = rval[2]
            pos_updates[self.ch] = T.cast(0.999 * self.ch + 0.001 * pos_h.mean(axis=0), floatX)

        return pos_updates

    def neg_sampling(self, n_steps=1):
        """
        Gibbs step for negative phase, which alternates:
        p(g|b,h,v), p(h|b,g,v), p(b|g,h,v), p(s|b,g,h,v) and p(v|b,g,h,s)
        :param g_sample: T.matrix of shape (batch_size, n_g)
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        :param n_steps: number of Gibbs updates to perform in negative phase.
        """

        def gibbs_iteration(g, t, h, s, v):
            new_h = self.sample_h_given_gtv(g, t, v)
            new_s = self.sample_s_given_gthv(g, t, new_h, v)
            new_g = self.sample_g_given_s(new_s)
            new_t = self.sample_t_given_gs(new_g, new_s)
            new_v = self.sample_v_given_hs(new_h, new_s)
            return [new_g, new_t, new_h, new_s, new_v]

        rvals , updates = theano.scan(
                gibbs_iteration,
                outputs_info = [self.neg_g, self.neg_t, self.neg_h, self.neg_s, self.neg_v],
                n_steps=n_steps)
        
        return [rval[-1] for rval in rvals]

    def neg_sampling_updates(self, n_steps=1):
        """
        Implements the negative phase, generating samples from p(h,s,v).
        :param n_steps: scalar, number of Gibbs steps to perform.
        """
        rval = self.neg_sampling(n_steps = n_steps)
        updates = OrderedDict()
        updates[self.neg_g] = rval[0]
        updates[self.neg_t] = rval[1]
        updates[self.neg_h] = rval[2]
        updates[self.neg_s] = rval[3]
        updates[self.neg_v] = rval[4]
        return updates

    def ml_cost(self, pos_g, pos_t1, pos_h, pos_s1, pos_s1_var, pos_s0, pos_s0_var, pos_v,
                      neg_g, neg_t, neg_h, neg_s, neg_v):
        """
        Variational approximation to the maximum likelihood positive phase.
        :param v: T.matrix of shape (batch_size, n_v), training examples
        :return: tuple (cost, gradient)
        """
        # L(q) = pos_cost + neg_cost + H(q)
        # pos_cost := E_{q(g)q(h)} log p(g,h,v)
        # neg_cost := -log Z
        pos_cost, pos_cte = self.lbound_plus(pos_g, pos_t1, pos_h,
                pos_s1, pos_s1_var,
                pos_s0, pos_s0_var, pos_v)

        # - dlogZ/dtheta = E_p[ denergy / dtheta ]
        neg_cost, neg_cte = self.energy(neg_g, neg_t, neg_h, neg_s, neg_v)

        # build gradient of cost with respect to model parameters
        cost = - (T.mean(pos_cost) + T.mean(neg_cost))
        cte = pos_cte + neg_cte

        return costmod.Cost(cost, self.params(), cte)


    ##############################
    # GENERIC OPTIMIZATION STUFF #
    ##############################
    def get_sparsity_cost(self, pos_g, pos_h):
        raise NotImplementedError()

    def get_reg_cost(self, l2=None, l1=None):
        """
        Builds the symbolic expression corresponding to first-order gradient descent
        of the cost function ``cost'', with some amount of regularization defined by the other
        parameters.
        :param l2: dict whose values represent amount of L2 regularization to apply to
        parameter specified by key.
        :param l1: idem for l1.
        """
        cost = T.zeros((), dtype=floatX)
        params = []

        for p in self.params():

            if l1.get(p.name, 0):
                cost += l1[p.name] * T.sum(abs(p))
                params += [p]

            if l2.get(p.name, 0):
                cost += l2[p.name] * T.sum(p**2)
                params += [p]
            
        return costmod.Cost(cost, params)

    def monitor_gauss(self, w, name=None):
        if name is None: assert hasattr(w, 'name')
        name = name if name else w.name

        rval = OrderedDict()
        rval[name + '.mean'] = w.mean()
        rval[name + '.std']  = w.std(axis=0).mean()
        rval[name + '.max']  = w.max()
        return rval

    def monitor_matrix(self, w, name=None, abs_mean=True):
        if name is None: assert hasattr(w, 'name')
        name = name if name else w.name

        rval = OrderedDict()
        rval[name + '.min'] = w.min(axis=[0,1])
        rval[name + '.max'] = w.max(axis=[0,1])
        if abs_mean:
            rval[name + '.absmean'] = abs(w).mean(axis=[0,1])
        else:
            rval[name + '.mean'] = w.mean(axis=[0,1])
        return rval

    def monitor_vector(self, b, name=None, abs_mean=True):
        if name is None: assert hasattr(b, 'name')
        name = name if name else b.name

        rval = OrderedDict()
        rval[name + '.min'] = b.min()
        rval[name + '.max'] = b.max()
        if abs_mean:
            rval[name + '.absmean'] = abs(b).mean()
        else:
            rval[name + '.mean'] = b.mean()
        return rval

    def get_monitoring_channels(self, x, y=None):
        chans = OrderedDict()
        chans.update(self.monitor_matrix(self.Wv))
        chans.update(self.monitor_matrix(self.Wg))
        chans.update(self.monitor_matrix(self.Wh))
        chans.update(self.monitor_vector(self.gbias))
        chans.update(self.monitor_vector(self.hbias))
        chans.update(self.monitor_vector(self.mu))
        chans.update(self.monitor_vector(self.nu))
        chans.update(self.monitor_vector(self.alpha_prec, name='alpha_prec'))
        chans.update(self.monitor_vector(self.beta_prec, name='beta_prec'))
        chans.update(self.monitor_vector(self.lambd_prec, name='lambd_prec'))
        chans.update(self.monitor_matrix(self.pos_g))
        chans.update(self.monitor_matrix(self.pos_h))
        chans.update(self.monitor_gauss(self.pos_t1))
        chans.update(self.monitor_gauss(self.pos_s1))
        chans.update(self.monitor_gauss(self.pos_s0))
        chans.update(self.monitor_matrix(self.neg_g))
        chans.update(self.monitor_matrix(self.neg_h))
        chans.update(self.monitor_gauss(self.neg_t))
        chans.update(self.monitor_gauss(self.neg_s))
        chans.update(self.monitor_gauss(self.neg_v))
        wg_norm = T.sqrt(T.sum(self.Wg**2, axis=0))
        wv_norm = T.sqrt(T.sum(self.Wv**2, axis=0))
        chans.update(self.monitor_vector(wg_norm, name='wg_norm'))
        chans.update(self.monitor_vector(wv_norm, name='wv_norm'))
        chans['lr'] = self.lr
        chans['pos_counter'] = self.pos_counter
        if self.flags['center_g']:
            chans.update(self.monitor_vector(self.cg))
        if self.flags['center_h']:
            chans.update(self.monitor_vector(self.ch))
        if self.flags['standardize_s']:
            chans.update(self.monitor_vector(self.gamma_s))
        if self.flags['standardize_t']:
            chans.update(self.monitor_vector(self.gamma_t))

        return chans

    def dlbound_dg(self, g, t1, h,
                   s1_mu, s1_mean, s1_var,
                   s0_mu, s0_mean, s0_var, v):

        lbound, cte = self.lbound(g, t1, t1, h,
                s1_mu, s1_mean, s1_var,
                s0_mu, s0_mean, s0_var, v, 
                t_stats_const=False)
        dlbound_dg = T.grad(T.sum(lbound), [g], consider_constant=cte)[0]
        return  g * (1-g) * dlbound_dg

    def dlbound_dh(self, g, t1, h,
                   s1_mu, s1_mean, s1_var,
                   s0_mu, s0_mean, s0_var, v):

        lbound, cte = self.lbound(g, t1, t1, h,
                s1_mu, s1_mean, s1_var,
                s0_mu, s0_mean, s0_var, v, 
                s_stats_const=False)
        dlbound_dh = T.grad(T.sum(lbound), [h], consider_constant=cte)[0]
        return h * (1-h) * dlbound_dh

    def init_debug(self):
        pass


import pylab as pl

class TrainingAlgorithm(default.DefaultTrainingAlgorithm):

    def init_params_from_data(self, model, x):

        if model.flags['ml_lambd']:
            # compute maximum likelihood solution for lambd
            scale = 1./(numpy.std(x, axis=0)**2)
            model.lambd.set_value(softplus_inv(scale).astype(floatX))
            # reset neg_v markov chain accordingly
            neg_v = model.rng.normal(loc=0, scale=scale, size=(model.batch_size, model.n_v))
            model.neg_v.set_value(neg_v.astype(floatX))

        if model.flags['pos_phase_ch']:
            model.inference_func(x[:model.batch_size])
            model.ch.set_value(model.pos_h.get_value().mean(axis=0))

    def setup(self, model, dataset):

        x = dataset.get_batch_design(10000, include_labels=False)
        self.init_params_from_data(model, x)
        super(TrainingAlgorithm, self).setup(model, dataset)
