"""
This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""
import numpy
import md5
import pickle
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import function, shared
from theano.sandbox import linalg

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


class BilinearSpikeSlabRBM(Model, Block):
    """Spike & Slab Restricted Boltzmann Machine (RBM)  """

    def load_params(self, model):
        fp = open(model)
        model = pickle.load(fp)
        fp.close()

        self.Wv.set_value(model.Wv.get_value())
        self.hbias.set_value(model.hbias.get_value())
        self.mu.set_value(model.mu.get_value())
        self.eta.set_value(model.eta.get_value())
        self.alpha.set_value(model.alpha.get_value())
        self.beta.set_value(model.beta.get_value())
        self.lambd.set_value(model.lambd.get_value())
        self.scalar_norms.set_value(model.scalar_norms.get_value())
        # sync negative phase particles
        self.neg_g.set_value(model.neg_g.get_value())
        self.neg_s.set_value(model.neg_s.get_value())
        self.neg_h.set_value(model.neg_h.get_value())
        self.neg_t.set_value(model.neg_t.get_value())
        self.neg_v.set_value(model.neg_v.get_value())
        self.neg_ev.set_value(model.neg_ev.get_value())
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
        self.vbound.set_value(model.vbound.get_value())

    def validate_flags(self, flags):
        flags.setdefault('truncate_s', False)
        if len(flags.keys()) != 1:
            raise NotImplementedError('One or more flags are currently not implemented.')

    def __init__(self, 
            numpy_rng = None, theano_rng = None,
            n_g=99, n_h=99, n_f=None, bw_g=3, bw_h=3, n_v=100, init_from=None,
            sparse_gmask = None, sparse_hmask = None,
            pos_mf_steps=1, pos_sample_steps=0, neg_sample_steps=1,
            lr=None, lr_timestamp=None, lr_mults = {},
            iscales={}, clip_min={}, clip_max={}, truncation_bounds={},
            l1 = {}, l2 = {}, orth_lambda=0.,
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
        assert lr is not None
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
        if self.n_f is None:
            assert n_g / bw_g == n_h / bw_h
            self.n_f = (n_g / bw_g) * (bw_g * bw_h)

        # allocate symbolic variable for input
        self.input = T.matrix('input')
        self.vbound = sharedX(truncation_bounds['v'], name='vbound')
        self.init_parameters()
        self.init_chains()

        # learning rate, with deferred 1./t annealing
        self.iter = sharedX(0.0, name='iter')

        if lr['type'] == 'anneal':
            num = lr['init'] * lr['start'] 
            denum = T.maximum(lr['start'], lr['slope'] * self.iter)
            self.lr = T.maximum(lr['floor'], num/denum) 
        elif lr['type'] == 'linear':
            lr_start = npy_floatX(lr['start'])
            lr_end   = npy_floatX(lr['end'])
            self.lr = lr_start + self.iter * (lr_end - lr_start) / npy_floatX(self.max_updates)
        else:
            raise ValueError('Incorrect value for lr[type]')

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

    def init_weight(self, iscale, shape, name, normalize=True, axis=0):
        value =  self.rng.normal(size=shape) * iscale
        if normalize:
            value /= numpy.sqrt(numpy.sum(value**2, axis=axis))
        return sharedX(value, name=name)

    def init_parameters(self):

        # marginal precision on visible units 
        self.lambd = sharedX(self.iscales['lambd'] * numpy.ones(self.n_v), name='lambd')

        # init scalar norm for each entry of Wv
        sn_val = self.iscales['scalar_norms'] * numpy.ones(self.n_f)
        self.scalar_norms = sharedX(sn_val, name='scalar_norms')

        # init weight matrices
        self.Wv = self.init_weight(1.0, (self.n_v, self.n_f), 'Wv')
        if self.sparse_gmask or self.sparse_hmask:
            assert self.sparse_gmask and self.sparse_hmask
            self.Wg = sharedX(self.sparse_gmask.mask * self.iscales.get('Wg', 1.0), name='Wg')
            self.Wh = sharedX(self.sparse_hmask.mask * self.iscales.get('Wh', 1.0), name='Wh')
        else:
            self.Wg = self.init_weight(1.0, (self.n_g, self.n_f), 'Wg')
            self.Wh = self.init_weight(1.0, (self.n_h, self.n_f), 'Wh')

        # bias parameters of g, h
        self.gbias = sharedX(self.iscales['gbias'] * numpy.ones(self.n_g), name='gbias')
        self.hbias = sharedX(self.iscales['hbias'] * numpy.ones(self.n_h), name='hbias')
        # mean (mu) and precision (alpha) parameters on s
        self.mu = sharedX(self.iscales['mu']  * numpy.ones(self.n_g), name='mu')
        self.alpha = sharedX(self.iscales['alpha'] * numpy.ones(self.n_g), name='alpha')
        # mean (eta) and precision (beta) parameters on t
        self.eta = sharedX(self.iscales['eta'] * numpy.ones(self.n_h), name='eta')
        self.beta  = sharedX(self.iscales['beta'] * numpy.ones(self.n_h), name='beta')

        # optional reparametrization of precision parameters
        self.lambd_prec = T.nnet.softplus(self.lambd)
        self.alpha_prec = T.nnet.softplus(self.alpha)
        self.beta_prec = T.nnet.softplus(self.beta)

    def init_chains(self):
        """ Allocate shared variable for persistent chain """
        self.neg_g  = sharedX(self.rng.rand(self.batch_size, self.n_g), name='neg_g')
        self.neg_s  = sharedX(self.rng.rand(self.batch_size, self.n_g), name='neg_s')
        self.neg_h  = sharedX(self.rng.rand(self.batch_size, self.n_h), name='neg_h')
        self.neg_t  = sharedX(self.rng.rand(self.batch_size, self.n_h), name='neg_t')
        self.neg_v  = sharedX(self.rng.rand(self.batch_size, self.n_v), name='neg_v')
        self.neg_ev = sharedX(self.rng.rand(self.batch_size, self.n_v), name='neg_ev')
 
    def params(self):
        """
        Returns a list of learnt model parameters.
        """
        params = [self.scalar_norms, self.Wv, self.lambd,
                  self.gbias, self.mu, self.alpha,
                  self.hbias, self.eta, self.beta]
        return params

    def do_theano(self):
        """ Compiles all theano functions needed to use the model"""

        init_names = dir(self)

        ###### All fields you don't want to get pickled (e.g., theano functions) should be created below this line
        # SAMPLING: NEGATIVE PHASE
        neg_updates = self.neg_sampling_updates(n_steps=self.neg_sample_steps, use_pcd=True)
        self.sample_func = theano.function([], [], updates=neg_updates)

        # VARIATIONAL E-STEP
        pos_updates = OrderedDict()
        if self.pos_mf_steps:
            pos_states, mf_updates = self.pos_phase_updates(
                    self.input,
                    mean_field = True,
                    n_steps = self.pos_mf_steps)
            pos_updates.update(mf_updates)

        # SAMPLING: POSITIVE PHASE
        if self.pos_sample_steps:
            init_state = pos_states if self.pos_mf_steps else None
            pos_states, sample_updates = self.pos_phase_updates(
                    self.input,
                    init_state = init_state,
                    mean_field = False,
                    n_steps = self.pos_sample_steps)
            pos_updates.update(sample_updates)

        ##
        # BUILD COST OBJECTS
        ##
        lcost = self.ml_cost(
                        pos_g = pos_states['g'],
                        pos_s = pos_states['s'],
                        pos_h = pos_states['h'],
                        pos_t = pos_states['t'],
                        pos_v = self.input,
                        neg_g = neg_updates[self.neg_g],
                        neg_s = neg_updates[self.neg_s],
                        neg_h = neg_updates[self.neg_h],
                        neg_t = neg_updates[self.neg_t],
                        neg_v = neg_updates[self.neg_v])

        spcost = self.get_sparsity_cost(
                pos_states['g'], pos_states['s'],
                pos_states['h'], pos_states['t'])

        regcost = self.get_reg_cost(self.l2, self.l1)

        ##
        # COMPUTE GRADIENTS WRT. COSTS
        ##
        main_cost = [lcost, spcost, regcost]
        learning_grads = costmod.compute_gradients(self.lr, self.lr_mults, *main_cost)

        weight_updates = OrderedDict()
        weight_updates[self.Wv] = true_gradient(self.Wv, -learning_grads[self.Wv])
        if self.Wg in self.params():
            weight_updates[self.Wg] = true_gradient(self.Wg, -learning_grads[self.Wg])
        if self.Wh in self.params():
            weight_updates[self.Wh] = true_gradient(self.Wh, -learning_grads[self.Wh])

        ##
        # BUILD UPDATES DICTIONARY FROM GRADIENTS
        ##
        learning_updates = costmod.get_updates(learning_grads)
        learning_updates.update(pos_updates)
        learning_updates.update(neg_updates)
        learning_updates.update({self.iter: self.iter+1})
        learning_updates.update(weight_updates)

        # build theano function to train on a single minibatch
        self.batch_train_func = function([self.input], [],
                                         updates=learning_updates,
                                         name='train_rbm_func')

        self.energy_fn = function([], self.energy(self.neg_g, self.neg_s, self.neg_h,
            self.neg_t, self.neg_v))

        self.g_fn = function([], self.g_given_htv(self.neg_h, self.neg_t, self.neg_v))
        self.h_fn = function([], self.h_given_gsv(self.neg_g, self.neg_s, self.neg_v))
        self.s_fn = function([], self.s_given_ghtv(self.neg_g, self.neg_h, self.neg_t, self.neg_v))
        self.t_fn = function([], self.t_given_gshv(self.neg_g, self.neg_s, self.neg_h, self.neg_v))
        self.v_fn = function([], self.v_given_gsht(self.neg_g, self.neg_s, self.neg_h, self.neg_t))
        self.sample_g_fn = function([], self.sample_g_given_htv(self.neg_h, self.neg_t, self.neg_v))
        self.sample_h_fn = function([], self.sample_h_given_gsv(self.neg_g, self.neg_s, self.neg_v))
        self.sample_s_fn = function([], self.sample_s_given_ghtv(self.neg_g, self.neg_h, self.neg_t, self.neg_v))
        self.sample_t_fn = function([], self.sample_t_given_gshv(self.neg_g, self.neg_s, self.neg_h, self.neg_v))
        self.sample_v_fn = function([], self.sample_v_given_gsht(self.neg_g, self.neg_s, self.neg_h, self.neg_t))

        #######################
        # CONSTRAINT FUNCTION #
        #######################

        # enforce constraints function
        constraint_updates = OrderedDict()
        constraint_updates[self.lambd] = T.mean(self.lambd) * T.ones_like(self.lambd)

        ## clip parameters to maximum values (if applicable)
        for (k,v) in self.clip_max.iteritems():
            assert k in [param.name for param in self.params()]
            param = getattr(self, k)
            constraint_updates[param] = T.clip(param, param, v)

        ## clip parameters to minimum values (if applicable)
        for (k,v) in self.clip_min.iteritems():
            assert k in [param.name for param in self.params()]
            param = getattr(self, k)
            constraint_updates[param] = T.clip(constraint_updates.get(param, param), v, param)
        
        self.enforce_constraints = theano.function([],[], updates=constraint_updates)

        ###### All fields you don't want to get pickled should be created above this line
        final_names = dir(self)
        self.register_names_to_del( [ name for name in (final_names) if name not in init_names ])

        # Before we start learning, make sure constraints are enforced
        self.enforce_constraints()

    def train_batch(self, dataset, batch_size):

        x = dataset.get_batch_design(batch_size, include_labels=False)
        vbound = self.vbound.get_value()
        x = numpy.clip(x, -vbound, vbound)
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


    def energy(self, g_sample, s_sample, h_sample, t_sample, v_sample):
        """
        Computes energy for a given configuration of (g,h,v,x,y).
        :param h_sample: T.matrix of shape (batch_size, n_g)
        :param s_sample: T.matrix of shape (batch_size, n_g)
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param t_sample: T.matrix of shape (batch_size, n_h)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        """
        from_g = self.from_g(g_sample * s_sample)
        from_h = self.from_h(h_sample * t_sample)
        from_v = self.from_v(v_sample)
        energy = -T.sum(from_g * from_h * from_v, axis=1)
        energy += T.sum(0.5 * self.alpha_prec * s_sample**2, axis=1)
        energy += T.sum(0.5 * self.beta_prec * t_sample**2, axis=1)
        energy -= T.sum(self.alpha_prec * self.mu * s_sample * g_sample, axis=1)
        energy -= T.sum(self.beta_prec * self.eta * t_sample * h_sample, axis=1)
        energy += T.sum(0.5 * self.alpha_prec * self.mu**2 * g_sample, axis=1)
        energy += T.sum(0.5 * self.beta_prec * self.eta**2 * h_sample, axis=1)
        energy += T.sum(0.5 * self.lambd_prec * v_sample**2, axis=1)
        energy -= T.dot(g_sample, self.gbias)
        energy -= T.dot(h_sample, self.hbias)
        return energy


    ######################################
    # MATH FOR CONDITIONAL DISTRIBUTIONS #
    ######################################

    def from_v(self, v_sample):
        Wv = self.scalar_norms * self.Wv
        return T.dot(self.lambd_prec * v_sample, Wv)

    def from_g(self, g_sample):
        return T.dot(g_sample, self.Wg)

    def from_h(self, h_sample):
        return T.dot(h_sample, self.Wh)

    def v_to_g(self, v_sample):
        return T.dot(self.from_v(v_sample), self.Wg.T)

    def v_to_h(self, v_sample):
        return T.dot(self.from_v(v_sample), self.Wh.T)
    
    def g_to_h(self, g_sample):
        return T.dot(self.from_g(g_sample), self.Wh.T)
    
    def h_to_g(self, h_sample):
        return T.dot(self.from_h(h_sample), self.Wg.T)

    def g_given_htv_input(self, h_sample, t_sample, v_sample):
        from_v = self.v_to_g(v_sample)
        from_h = self.h_to_g(h_sample * t_sample)
        g_mean  = 0.5 * 1./self.alpha_prec * (from_h * from_v)**2
        g_mean += self.mu * (from_h * from_v)
        g_mean += self.gbias
        return g_mean
    
    def g_given_htv(self, h_sample, t_sample, v_sample):
        g_mean = self.g_given_htv_input(h_sample, t_sample, v_sample)
        return T.nnet.sigmoid(g_mean)

    def sample_g_given_htv(self, h_sample, t_sample, v_sample, rng=None):
        """
        Generates sample from p(g | h, t, v)
        """
        g_mean = self.g_given_htv(h_sample, t_sample, v_sample)

        rng = self.theano_rng if rng is None else rng
        g_sample = rng.binomial(size=(self.batch_size,self.n_g),
                                n=1, p=g_mean, dtype=floatX)
        return g_sample

    def s_given_ghtv(self, g_sample, h_sample, t_sample, v_sample):
        from_h = self.h_to_g(h_sample * t_sample)
        from_v = self.v_to_g(v_sample)
        s_mean = (1./self.alpha_prec * from_v * from_h + self.mu) * g_sample
        return s_mean

    def sample_s_given_ghtv(self, g_sample, h_sample, t_sample, v_sample, rng=None):
        s_mean = self.s_given_ghtv(g_sample, h_sample, t_sample, v_sample)
        
        rng = self.theano_rng if rng is None else rng

        if self.flags['truncate_s']:
            s_sample = truncated.truncated_normal(
                    size=(self.batch_size, self.n_g),
                    avg = s_mean, 
                    std = T.sqrt(1./self.alpha_prec),
                    lbound = self.mu - self.truncation_bounds['s'],
                    ubound = self.mu + self.truncation_bounds['s'],
                    theano_rng = rng,
                    dtype=floatX)
        else: 
            s_sample = rng.normal(
                    size=(self.batch_size, self.n_g),
                    avg = s_mean, 
                    std = T.sqrt(1./self.alpha_prec), dtype=floatX)
        return s_sample

    def h_given_gsv_input(self, g_sample, s_sample, v_sample):
        from_v = self.v_to_h(v_sample)
        from_g = self.g_to_h(g_sample * s_sample)
        h_mean  = 0.5 * 1./self.beta_prec * (from_g * from_v)**2
        h_mean += self.eta * (from_g * from_v)
        h_mean += self.hbias
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

    def t_given_gshv(self, g_sample, s_sample, h_sample, v_sample):
        from_g = self.g_to_h(g_sample * s_sample)
        from_v = self.v_to_h(v_sample)
        t_mean = (1./self.beta_prec * from_v * from_g + self.eta) * h_sample
        return t_mean

    def sample_t_given_gshv(self, g_sample, s_sample, h_sample, v_sample, rng=None):
        t_mean = self.t_given_gshv(g_sample, s_sample, h_sample, v_sample)
        
        rng = self.theano_rng if rng is None else rng

        if self.flags['truncate_s']:
            t_sample = truncated.truncated_normal(
                    size=(self.batch_size, self.n_h),
                    avg = t_mean, 
                    std = T.sqrt(1./self.beta_prec),
                    lbound = self.eta - self.truncation_bounds['t'],
                    ubound = self.eta + self.truncation_bounds['t'],
                    theano_rng = rng,
                    dtype=floatX)
        else: 
            t_sample = rng.normal(
                    size=(self.batch_size, self.n_h),
                    avg = t_mean, 
                    std = T.sqrt(1./self.beta_prec), dtype=floatX)
        return t_sample


    def v_given_gsht(self, g_sample, s_sample, h_sample, t_sample):
        Wv = self.scalar_norms * self.Wv
        from_g = self.from_g(g_sample * s_sample)
        from_h = self.from_h(h_sample * t_sample)
        v_mean = T.dot(from_g * from_h, Wv.T)
        return v_mean

    def sample_v_given_gsht(self, g_sample, s_sample, h_sample, t_sample, rng=None):
        v_mean = self.v_given_gsht(g_sample, s_sample, h_sample, t_sample)

        rng = self.theano_rng if rng is None else rng
        v_sample = truncated.truncated_normal(
                size=(self.batch_size, self.n_v),
                avg = v_mean, 
                std = T.sqrt(1./self.lambd_prec),
                lbound = -self.vbound,
                ubound = self.vbound,
                theano_rng = rng,
                dtype=floatX)
        return v_sample

    ##################
    # SAMPLING STUFF #
    ##################
    def pos_phase(self, v, init_state, n_steps=1, mean_field=False):
        """
        Mixed mean-field + sampling inference in positive phase.
        :param v: input being conditioned on
        :param init: dictionary of initial values
        :param n_steps: number of Gibbs updates to perform afterwards.
        """
        def gibbs_iteration(g1, s1, h1, t1, v):
            if mean_field:
                g2 = self.g_given_htv(h1, t1, v) 
                s2 = self.s_given_ghtv(T.ones_like(g2), h1, t1, v)
                h2 = self.h_given_gsv(g2, s2, v)
                t2 = self.t_given_gshv(g2, s2, T.ones_like(h2), v)
            else:
                g2 = self.sample_g_given_htv(h1, t1, v) 
                s2 = self.sample_s_given_ghtv(g2, h1, t1, v)
                h2 = self.sample_h_given_gsv(g2, s2, v)
                t2 = self.sample_t_given_gshv(g2, s2, h2, v)
            return [g2, s2, h2, t2]

        [new_g, new_s, new_h, new_t], updates = theano.scan(
                gibbs_iteration,
                outputs_info = [init_state['g'], init_state['s'],
                                init_state['h'], init_state['t']],
                non_sequences = [v],
                n_steps=n_steps)

        new_g = new_g[-1]
        new_s = new_s[-1]
        new_h = new_h[-1]
        new_t = new_t[-1]

        return [new_g, new_s, new_h, new_t]

    def pos_phase_updates(self, v, init_state=None, n_steps=1, mean_field=False):
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
            init_state['g'] = T.ones((self.batch_size,self.n_g)) * T.nnet.sigmoid(self.gbias)
            init_state['s'] = T.ones((self.batch_size,self.n_g)) * self.mu
            init_state['h'] = T.ones((self.batch_size,self.n_h)) * T.nnet.sigmoid(self.hbias)
            init_state['t'] = T.ones((self.batch_size,self.n_h)) * self.eta

        [new_g, new_s, new_h, new_t] = self.pos_phase(v,
                init_state = init_state,
                n_steps = n_steps,
                mean_field = mean_field)

        pos_states = OrderedDict()
        pos_states['g'] = new_g
        pos_states['s'] = new_s
        pos_states['h'] = new_h
        pos_states['t'] = new_t

        # update running average of positive phase activations
        pos_updates = OrderedDict()
        return pos_states, pos_updates

    def neg_sampling(self, g_sample, s_sample, h_sample, t_sample, v_sample, n_steps=1):
        """
        Gibbs step for negative phase, which alternates:
        p(g|b,h,v), p(h|b,g,v), p(b|g,h,v), p(s|b,g,h,v) and p(v|b,g,h,s)
        :param g_sample: T.matrix of shape (batch_size, n_g)
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param s_sample: T.matrix of shape (batch_size, n_s)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        :param n_steps: number of Gibbs updates to perform in negative phase.
        """
        def gibbs_iteration(g1, s1, h1, t1, v1):
            g2 = self.sample_g_given_htv(h1, t1, v1)
            s2 = self.sample_s_given_ghtv(g2, h1, t1, v1)
            h2 = self.sample_h_given_gsv(g2, s2, v1)
            t2 = self.sample_t_given_gshv(g2, s2, h2, v1)
            v2 = self.sample_v_given_gsht(g2, s2, h2, t2)
            return [g2, s2, h2, t2, v2]

        [new_g, new_s, new_h, new_t, new_v] , updates = theano.scan(
                gibbs_iteration,
                outputs_info = [g_sample, s_sample, h_sample, t_sample, v_sample],
                n_steps=n_steps)

        return [new_g[-1], new_s[-1], new_h[-1], new_t[-1], new_v[-1]]

    def neg_sampling_updates(self, n_steps=1, use_pcd=True):
        """
        Implements the negative phase, generating samples from p(h,s,v).
        :param n_steps: scalar, number of Gibbs steps to perform.
        """
        init_chain = self.neg_v if use_pcd else self.input
        [new_g, new_s, new_h, new_t, new_v] =  self.neg_sampling(
                self.neg_g, self.neg_s,
                self.neg_h, self.neg_t,
                self.neg_v, n_steps = n_steps)

        # we want to plot the expected value of the samples
        new_ev = self.v_given_gsht(new_g, new_s, new_h, new_t)

        updates = OrderedDict()
        updates[self.neg_g] = new_g
        updates[self.neg_s] = new_s
        updates[self.neg_h] = new_h
        updates[self.neg_t] = new_t
        updates[self.neg_v] = new_v
        updates[self.neg_ev] = new_ev
        return updates

    def ml_cost(self, pos_g, pos_s, pos_h, pos_t, pos_v,
                      neg_g, neg_s, neg_h, neg_t, neg_v):

        pos_cost = T.sum(self.energy(pos_g, pos_s, pos_h, pos_t, pos_v))
        neg_cost = T.sum(self.energy(neg_g, neg_s, neg_h, neg_t, neg_v))
        batch_cost = pos_cost - neg_cost
        cost = batch_cost / self.batch_size

        # build gradient of cost with respect to model parameters
        cte = [pos_g, pos_s, pos_h, pos_t, pos_v,
               neg_g, neg_s, neg_h, neg_t, neg_v]

        return costmod.Cost(cost, self.params(), cte)

    ##############################
    # GENERIC OPTIMIZATION STUFF #
    ##############################
    def get_sparsity_cost(self, pos_g, pos_s, pos_h, pos_t):

        # update mean activation using exponential moving average
        hack_g = self.g_given_htv(pos_h, pos_t, self.input)
        hack_h = self.h_given_gsv(pos_g, pos_s, self.input)

        # define loss based on value of sp_type
        eps = npy_floatX(1./self.batch_size)
        loss = lambda targ, val: - npy_floatX(targ) * T.log(eps + val) \
                                 - npy_floatX(1-targ) * T.log(1 - val + eps)

        params = []
        cost = T.zeros((), dtype=floatX)
        if self.sp_weight['g'] or self.sp_weight['h']:
            params += [self.scalar_norms, self.Wv]
            if self.sp_weight['g']:
                cost += self.sp_weight['g']  * T.sum(loss(self.sp_targ['g'], hack_g.mean(axis=0)))
                params += [self.alpha, self.mu, self.gbias]
            if self.sp_weight['h']:
                cost += self.sp_weight['h']  * T.sum(loss(self.sp_targ['h'], hack_h.mean(axis=0)))
                params += [self.beta, self.eta, self.hbias]

        cte = [pos_g, pos_s, pos_h, pos_t]
        return costmod.Cost(cost, params, cte)

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

    def monitor_matrix(self, w, name=None):
        if name is None: assert hasattr(w, 'name')
        name = name if name else w.name

        return {name + '.min':  w.min(axis=[0,1]),
                name + '.max':  w.max(axis=[0,1]),
                name + '.absmean': abs(w).mean(axis=[0,1])}

    def monitor_vector(self, b, name=None):
        if name is None: assert hasattr(b, 'name')
        name = name if name else b.name

        return {name + '.min':  b.min(),
                name + '.max':  b.max(),
                name + '.absmean': abs(b).mean()}

    def get_monitoring_channels(self, x, y=None):
        chans = OrderedDict()
        chans.update(self.monitor_vector(self.scalar_norms))
        chans.update(self.monitor_matrix(self.Wv))
        chans.update(self.monitor_matrix(self.Wg))
        chans.update(self.monitor_matrix(self.Wh))
        chans.update(self.monitor_vector(self.gbias))
        chans.update(self.monitor_vector(self.hbias))
        chans.update(self.monitor_vector(self.mu))
        chans.update(self.monitor_vector(self.eta))
        chans.update(self.monitor_vector(self.alpha_prec, name='alpha_prec'))
        chans.update(self.monitor_vector(self.beta_prec, name='beta_prec'))
        chans.update(self.monitor_vector(self.lambd_prec, name='lambda_prec'))
        chans.update(self.monitor_matrix(self.neg_g))
        chans.update(self.monitor_matrix(self.neg_s - self.mu, name='(neg_s - mu)'))
        chans.update(self.monitor_matrix(self.neg_h))
        chans.update(self.monitor_matrix(self.neg_t - self.eta, name='(neg_t - eta)'))
        chans.update(self.monitor_matrix(self.neg_v))
        wg_norm = T.sqrt(T.sum(self.Wg**2, axis=0))
        wh_norm = T.sqrt(T.sum(self.Wh**2, axis=0))
        wv_norm = T.sqrt(T.sum(self.Wv**2, axis=0))
        chans.update(self.monitor_vector(wg_norm, name='wg_norm'))
        chans.update(self.monitor_vector(wh_norm, name='wh_norm'))
        chans.update(self.monitor_vector(wv_norm, name='wv_norm'))
        chans['lr'] = self.lr
        return chans


class FactoredBilinearSpikeSlabRBM(BilinearSpikeSlabRBM):
 
    def params(self):
        """
        Returns a list of learnt model parameters.
        """
        params = [self.Wg, self.Wh, self.Wv,
                  self.scalar_norms, self.lambd,
                  self.gbias, self.mu, self.alpha,
                  self.hbias, self.eta, self.beta]
        return params


class TrainingAlgorithm(default.DefaultTrainingAlgorithm):

    def setup(self, model, dataset):
        super(TrainingAlgorithm, self).setup(model, dataset)
