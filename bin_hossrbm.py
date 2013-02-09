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

import cost as costmod
from utils import tools
from utils import rbm_utils
from utils import sharedX, floatX, npy_floatX
from true_gradient import true_gradient


class BinaryBilinearSpikeSlabRBM(Model, Block):
    """Spike & Slab Restricted Boltzmann Machine (RBM)  """

    def load_params(self, model):
        fp = open(model)
        model = pickle.load(fp)
        fp.close()

        self.Wv.set_value(model.Wv.get_value())
        self.hbias.set_value(model.hbias.get_value())
        self.mu.set_value(model.mu.get_value())
        self.alpha.set_value(model.alpha.get_value())
        self.scalar_norms.set_value(model.scalar_norms.get_value())
        # sync negative phase particles
        self.neg_v.set_value(model.neg_v.get_value())
        self.neg_s.set_value(model.neg_s.get_value())
        self.neg_h.set_value(model.neg_h.get_value())
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

    def validate_flags(self, flags):
        flags.setdefault('wg_norm', 'none')
        flags.setdefault('wh_norm', 'none')
        flags.setdefault('wv_norm', 'none')
        flags.setdefault('split_norm', False)
        flags.setdefault('mean_field', False)
        if len(flags.keys()) != 5:
            raise NotImplementedError('One or more flags are currently not implemented.')

    @classmethod
    def init_from_model(cls, model):
        return BinaryBilinearSpikeSlabRBM(
                refmodel = model,
                n_g=model.n_g,
                n_h=model.n_h,
                n_s=model.n_s,
                n_v=model.n_v,
                sparse_gmask=model.sparse_gmask,
                sparse_hmask=model.sparse_hmask,
                pos_steps=model.pos_steps,
                iscales=model.iscales,
                clip_min=model.clip_min,
                clip_max=model.clip_max,
                lr_spec=model.lr_spec,
                lr_timestamp=model.lr_timestamp,
                lr_mults=model.lr_mults,
                l1=model.l1,
                l2=model.l2,
                sp_weight=model.sp_weight,
                sp_targ=model.sp_targ,
                batch_size=model.batch_size,
                my_save_path=model.my_save_path + '_nolabel',
                save_at=model.save_at,
                save_every=model.save_every,
                flags=model.flags,
                max_updates=model.max_updates)

    def __init__(self, 
            numpy_rng = None, theano_rng = None, refmodel=None,
            n_g=99, n_h=99, n_s=99, n_v=100, init_from=None,
            sparse_gmask = None, sparse_hmask = None,
            pos_steps=1, neg_sample_steps=1,
            lr_spec=None, lr_timestamp=None, lr_mults = {},
            iscales={}, clip_min={}, clip_max={},
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
        if refmodel: 
            self.init_parameters_from_model(refmodel)
        else:
            self.init_parameters()
        self.init_chains()

        # learning rate, with deferred 1./t annealing
        self.iter = sharedX(0.0, name='iter')

        if lr_spec['type'] == 'anneal':
            num = lr_spec['init'] * lr_spec['start'] 
            pos = T.maximum(lr_spec['start'], lr_spec['slope'] * self.iter)
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

    def init_weight(self, iscale, shape, name, normalize=True, axis=0):
        value =  self.rng.normal(size=shape) * iscale
        if normalize:
            value /= numpy.sqrt(numpy.sum(value**2, axis=axis))
        return sharedX(value, name=name)

    def init_parameters_from_model(self, model):
        self.scalar_norms = model.scalar_norms
        self.Wv = model.Wv
        self.Wg = model.Wg
        self.Wh = model.Wh
        self.avg_norm_wg = model.avg_norm_wg
        self.avg_norm_wh = model.avg_norm_wh
        self.gbias = model.gbias
        self.hbias = sharedX(self.iscales['hbias'] * numpy.ones(self.n_h), name='hbias')
        self.vbias = model.vbias
        self.mu = model.mu
        self.alpha = model.alpha
        self.alpha_prec = model.alpha_prec

    def init_parameters(self):
        # init scalar norm for each entry of Wv
        sn_val = self.iscales['scalar_norms'] * numpy.ones(self.n_s)
        self.scalar_norms = sharedX(sn_val, name='scalar_norms')

        # init weight matrices
        normalize_wv = self.flags['wv_norm'] == 'unit' 
        self.Wv = self.init_weight(self.iscales['Wv'], (self.n_v, self.n_s), 'Wv', normalize=normalize_wv)
        if self.sparse_gmask or self.sparse_hmask:
            assert self.sparse_gmask and self.sparse_hmask
            self.Wg = sharedX(self.sparse_gmask.mask * self.iscales.get('Wg', 1.0), name='Wg')
            self.Wh = sharedX(self.sparse_hmask.mask * self.iscales.get('Wh', 1.0), name='Wh')
        else:
            normalize_wg = self.flags['wg_norm'] == 'unit'
            normalize_wh = self.flags['wh_norm'] == 'unit'
            self.Wg = self.init_weight(self.iscales['Wg'], (self.n_g, self.n_s), 'Wg', normalize=normalize_wg)
            self.Wh = self.init_weight(self.iscales['Wh'], (self.n_h, self.n_s), 'Wh', normalize=normalize_wh)

        # avg norm (for wgh_norm='roland')
        norm_wg = numpy.sqrt(numpy.sum(self.Wg.get_value()**2, axis=0)).mean()
        norm_wh = numpy.sqrt(numpy.sum(self.Wh.get_value()**2, axis=0)).mean()
        self.avg_norm_wg = sharedX(norm_wg, name='avg_norm_wg')
        self.avg_norm_wh = sharedX(norm_wh, name='avg_norm_wh')

        # allocate shared variables for bias parameters
        self.gbias = sharedX(self.iscales['gbias'] * numpy.ones(self.n_g), name='gbias')
        self.hbias = sharedX(self.iscales['hbias'] * numpy.ones(self.n_h), name='hbias')
        self.vbias = sharedX(self.iscales['vbias'] * numpy.ones(self.n_v), name='vbias')

        # mean (mu) and precision (alpha) parameters on s
        self.mu = sharedX(self.iscales['mu'] * numpy.ones(self.n_s), name='mu')
        self.alpha = sharedX(self.iscales['alpha'] * numpy.ones(self.n_s), name='alpha')
        self.alpha_prec = T.nnet.softplus(self.alpha)

    def init_chains(self):
        """ Allocate shared variable for persistent chain """
        # initialize binary g-h chains
        neg_g = self.rng.binomial(n=1, p=self.gbias.get_value(), size=(self.batch_size, self.n_g))
        neg_h = self.rng.binomial(n=1, p=self.hbias.get_value(), size=(self.batch_size, self.n_h))
        neg_v = self.rng.binomial(n=1, p=self.vbias.get_value(), size=(self.batch_size, self.n_v))
        self.neg_g  = sharedX(neg_g, name='neg_g')
        self.neg_h  = sharedX(neg_h, name='neg_h')
        self.neg_v  = sharedX(neg_v, name='neg_v')
        # initialize s-chain
        def softplus(x): return numpy.log(1. + numpy.exp(x))
        loc = self.mu.get_value()
        scale = numpy.sqrt(1./softplus(self.alpha.get_value()))
        neg_s  = self.rng.normal(loc=loc, scale=scale, size=(self.batch_size, self.n_s))
        self.neg_s  = sharedX(neg_s, name='neg_s')
 
    def params(self):
        """
        Returns a list of learnt model parameters.
        """
        params = [self.Wv, self.hbias, self.gbias, self.vbias, self.alpha, self.mu]
        if self.flags['split_norm']:
            params += [self.scalar_norms]
        return params

    def do_theano(self):
        """ Compiles all theano functions needed to use the model"""

        init_names = dir(self)

        ###### All fields you don't want to get pickled (e.g., theano functions) should be created below this line
        # SAMPLING: NEGATIVE PHASE
        neg_updates = self.neg_sampling_updates(n_steps=self.neg_sample_steps, use_pcd=True)
        self.sample_func = theano.function([], [], updates=neg_updates)

        # POSITIVE PHASE
        pos_states, pos_updates = self.pos_phase_updates(
                self.input,
                n_steps = self.pos_steps,
                mean_field=self.flags['mean_field'])

        ##
        # BUILD COST OBJECTS
        ##
        lcost = self.ml_cost(
                        pos_g = pos_states['g'],
                        pos_h = pos_states['h'],
                        pos_v = self.input,
                        neg_g = neg_updates[self.neg_g],
                        neg_h = neg_updates[self.neg_h],
                        neg_v = neg_updates[self.neg_v],
                        mean_field=self.flags['mean_field'])
        spcost = self.get_sparsity_cost(pos_states['g'], pos_states['h'])
        regcost = self.get_reg_cost(self.l2, self.l1)

        ##
        # COMPUTE GRADIENTS WRT. COSTS
        ##
        main_cost = [lcost, spcost, regcost]

        learning_grads = costmod.compute_gradients(self.lr, self.lr_mults, *main_cost)

        weight_updates = OrderedDict()
        if self.flags['wg_norm'] == 'unit' and self.Wg in self.params():
            weight_updates[self.Wg] = true_gradient(self.Wg, -learning_grads[self.Wg])
        if self.flags['wh_norm'] == 'unit' and self.Wh in self.params():
            weight_updates[self.Wh] = true_gradient(self.Wh, -learning_grads[self.Wh])
        if self.flags['wv_norm'] == 'unit':
            weight_updates[self.Wv] = true_gradient(self.Wv, -learning_grads[self.Wv])

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

    def get_constraint_updates(self):
        constraint_updates = OrderedDict() 
        if self.Wg in self.params():
            norm_wg = T.sqrt(T.sum(self.Wg**2, axis=0))
            if self.flags['wg_norm'] == 'roland':
                constraint_updates[self.Wg] = self.avg_norm_wg * self.Wg / norm_wg
                constraint_updates[self.avg_norm_wg] = 0.9 * self.avg_norm_wg + 0.1 * T.mean(norm_wg)
            elif self.flags['wg_norm'] == 'max_unit':
                constraint_updates[self.Wg] = self.Wg / norm_wg * T.minimum(norm_wg, 1.0)

        if self.Wh in self.params():
            norm_wh = T.sqrt(T.sum(self.Wh**2, axis=0))
            if self.flags['wh_norm'] == 'roland':
                constraint_updates[self.Wh] = self.avg_norm_wh * self.Wh / norm_wh
                constraint_updates[self.avg_norm_wh] = 0.9 * self.avg_norm_wh + 0.1 * T.mean(norm_wh)
            elif self.flags['wh_norm'] == 'max_unit':
                constraint_updates[self.Wh] = self.Wh / norm_wh * T.minimum(norm_wh, 1.0)

        if self.flags['wv_norm'] == 'max_unit':
            norm_wv = T.sqrt(T.sum(self.Wv**2, axis=0))
            constraint_updates[self.Wv] = self.Wv / norm_wv * T.minimum(norm_wv, 1.0)
            constraint_updates[self.scalar_norms] = T.maximum(1.0, self.scalar_norms)

        ## clip parameters to maximum values (if applicable)
        for (k,v) in self.clip_max.iteritems():
            assert k in [param.name for param in self.params()]
            param = constraint_updates.get(k, getattr(self, k))
            constraint_updates[param] = T.clip(param, param, v)

        ## clip parameters to minimum values (if applicable)
        for (k,v) in self.clip_min.iteritems():
            assert k in [param.name for param in self.params()]
            param = constraint_updates.get(k, getattr(self, k))
            constraint_updates[param] = T.clip(constraint_updates.get(param, param), v, param)

        return constraint_updates

    def train_batch(self, dataset, batch_size):

        x = dataset.get_batch_design(batch_size, include_labels=False)
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

    def free_energy(self, g_sample, h_sample, v_sample):
        """
        Computes energy for a given configuration of (g,h,v,x,y).
        :param g_sample: T.matrix of shape (batch_size, n_g)
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        """
        from_v = self.from_v(v_sample)
        from_h = self.from_h(h_sample)
        from_g = self.from_g(g_sample)
        energy = -T.sum(0.5 * 1./self.alpha_prec * from_g * from_h * from_v**2, axis=1)
        energy -= T.sum(self.mu * from_g * from_h * from_v, axis=1)
        energy -= 0.5 * T.sum(T.log(2*numpy.pi / self.alpha_prec))
        energy -= T.dot(g_sample, self.gbias)
        energy -= T.dot(h_sample, self.hbias)
        energy -= T.dot(v_sample, self.vbias)
        return T.mean(energy), [g_sample, h_sample, v_sample]



    def __call__(self, v, output_type='g+h', mean_field=True):
        print 'Building representation with %s' % output_type
        init_state = OrderedDict()
        init_state['g'] = T.ones((v.shape[0],self.n_g)) * T.nnet.sigmoid(self.gbias)
        init_state['h'] = T.ones((v.shape[0],self.n_h)) * T.nnet.sigmoid(self.hbias)
        [g, h] = self.pos_phase(v, init_state, n_steps=self.pos_steps, mean_field=mean_field)

        atoms = {
                'g_s' : T.dot(g, self.Wg),  # g in s-space
                'h_s' : T.dot(h, self.Wh),  # h in s-space
                }

        output_prods = {
                'g' : g,
                'h' : h,
                'gh' : (g.dimshuffle(0,1,'x') * h.dimshuffle(0,'x',1)).flatten(ndim=2),
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
        Wv = self.Wv
        if self.flags['split_norm']:
            Wv *= self.scalar_norms
        return T.dot(v_sample, Wv)

    def from_g(self, g_sample):
        return T.dot(g_sample, self.Wg)

    def from_h(self, h_sample):
        return T.dot(h_sample, self.Wh)

    def to_g(self, g_s):
        return T.dot(g_s, self.Wg.T) + self.gbias

    def to_h(self, h_s):
        return T.dot(h_s, self.Wh.T) + self.hbias

    def h_given_gv_input(self, g_sample, v_sample):
        """
        Compute mean activation of h given v.
        :param g_sample: T.matrix of shape (batch_size, n_g matrix)
        :param v_sample: T.matrix of shape (batch_size, n_v matrix)
        """
        from_v = self.from_v(v_sample)
        from_g = self.from_g(g_sample)
        h_mean_s = 0.5 * 1./self.alpha_prec * from_g * from_v**2
        h_mean_s += from_g * from_v * self.mu
        h_mean = self.to_h(h_mean_s)
        return h_mean
    
    def h_given_gv(self, g_sample, v_sample):
        h_mean = self.h_given_gv_input(g_sample, v_sample)
        return T.nnet.sigmoid(h_mean)

    def sample_h_given_gv(self, g_sample, v_sample, rng=None, size=None):
        """
        Generates sample from p(h | g, v)
        """
        h_mean = self.h_given_gv(g_sample, v_sample)

        rng = self.theano_rng if rng is None else rng
        size = size if size else self.batch_size
        h_sample = rng.binomial(size=(size, self.n_h),
                                n=1, p=h_mean, dtype=floatX)
        return h_sample

    def g_given_hv_input(self, h_sample, v_sample):
        """
        Compute mean activation of h given v.
        :param h_sample: T.matrix of shape (batch_size, n_h matrix)
        :param v_sample: T.matrix of shape (batch_size, n_v matrix)
        """
        from_v = self.from_v(v_sample)
        from_h = self.from_h(h_sample)
        g_mean_s = 0.5 * 1./self.alpha_prec * from_h * from_v**2
        g_mean_s += from_h * from_v * self.mu
        g_mean = self.to_g(g_mean_s)
        return g_mean
    
    def g_given_hv(self, h_sample, v_sample):
        g_mean = self.g_given_hv_input(h_sample, v_sample)
        return T.nnet.sigmoid(g_mean)

    def sample_g_given_hv(self, h_sample, v_sample, rng=None, size=None):
        """
        Generates sample from p(g | h, v)
        """
        g_mean = self.g_given_hv(h_sample, v_sample)

        rng = self.theano_rng if rng is None else rng
        size = size if size else self.batch_size
        g_sample = rng.binomial(size=(size, self.n_g),
                                n=1, p=g_mean, dtype=floatX)
        return g_sample

    def s_given_ghv(self, g_sample, h_sample, v_sample):
        from_g = self.from_g(g_sample)
        from_h = self.from_h(h_sample)
        from_v = self.from_v(v_sample)
        s_mean = (1./self.alpha_prec * from_v + self.mu) * from_g * from_h
        return s_mean

    def sample_s_given_ghv(self, g_sample, h_sample, v_sample, rng=None, size=None):
        s_mean = self.s_given_ghv(g_sample, h_sample, v_sample)
        
        rng = self.theano_rng if rng is None else rng
        size = size if size else self.batch_size
        s_sample = rng.normal(
                size=(size, self.n_s),
                avg = s_mean, 
                std = T.sqrt(1./self.alpha_prec),
                dtype=floatX)
        return s_sample

    def v_given_ghs(self, g_sample, h_sample, s_sample):
        """
        Computes the mean-activation of visible units, given all other variables.
        :param g_sample: T.matrix of shape (batch_size, n_g)
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param s_sample: T.matrix of shape (batch_size, n_s)
        """
        Wv = self.Wv
        if self.flags['split_norm']:
            Wv *= self.scalar_norms
        from_g = self.from_g(g_sample)
        from_h = self.from_h(h_sample)
        v_mean = T.dot(s_sample * from_g * from_h, Wv.T) + self.vbias
        return v_mean

    def sample_v_given_ghs(self, g_sample, h_sample, s_sample, rng=None, size=None):
        v_mean = self.v_given_ghs(g_sample, h_sample, s_sample)

        rng = self.theano_rng if rng is None else rng
        size = size if size else self.batch_size
        v_sample = rng.binomial(size=(size, self.n_v),
                                n=1, p=v_mean, dtype=floatX)
        return v_sample

    ##################
    # SAMPLING STUFF #
    ##################
    def pos_phase(self, v, init_state, n_steps=1, mean_field=False, eps=1e-3):
        """
        Mixed mean-field + sampling inference in positive phase.
        :param v: input being conditioned on
        :param init: dictionary of initial values
        :param n_steps: number of Gibbs updates to perform afterwards.
        """

        def pos_gibbs_iteration(g1, h1, v, size):
            g2 = self.sample_g_given_hv(h1, v, size=size)
            h2 = self.sample_h_given_gv(g2, v, size=size)
            return [g2, h2]

        def pos_mf_iteration(g1, h1, v, size):
            g2 = self.g_given_hv(h1, v)
            h2 = self.h_given_gv(g2, v)
            return [g2, h2], theano.scan_module.until(
                    (0.5 * (abs(g2 - g1).mean() + abs(h2-h1).mean()) < eps))

        [new_g, new_h], updates = theano.scan(
                pos_mf_iteration if mean_field else pos_gibbs_iteration,
                outputs_info = [init_state['g'], init_state['h']],
                non_sequences = [v, v.shape[0]],
                n_steps=n_steps)

        new_g = new_g[-1]
        new_h = new_h[-1]

        return [new_g, new_h]

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
            init_state['h'] = T.ones((self.batch_size,self.n_h)) * T.nnet.sigmoid(self.hbias)

        [new_g, new_h] = self.pos_phase(v,
                init_state=init_state,
                n_steps=n_steps,
                mean_field=mean_field)

        pos_states = OrderedDict()
        pos_states['g'] = new_g
        pos_states['h'] = new_h

        # update running average of positive phase activations
        pos_updates = OrderedDict()
        return pos_states, pos_updates

    def neg_sampling(self, g_sample, h_sample, v_sample, n_steps=1):
        """
        Gibbs step for negative phase, which alternates:
        p(g|b,h,v), p(h|b,g,v), p(b|g,h,v), p(s|b,g,h,v) and p(v|b,g,h,s)
        :param g_sample: T.matrix of shape (batch_size, n_g)
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        :param n_steps: number of Gibbs updates to perform in negative phase.
        """

        def gibbs_iteration(g1, h1, v1):
            g2 = self.sample_g_given_hv(h1, v1)
            h2 = self.sample_h_given_gv(g2, v1)
            s2 = self.sample_s_given_ghv(g2, h2, v1)
            v2 = self.sample_v_given_ghs(g2, h2, s2)
            return [g2, h2, s2, v2]

        [new_g, new_h, new_s, new_v] , updates = theano.scan(
                gibbs_iteration,
                outputs_info = [g_sample, h_sample, None, v_sample],
                n_steps=n_steps)

        final_v = new_v[-1]
        final_g = self.sample_g_given_hv(new_h[-1], final_v)
        final_h = self.sample_h_given_gv(final_g, final_v)
        final_s = self.sample_s_given_ghv(final_g, final_h, final_v)

        return [final_g, final_h, final_s, final_v]

    def neg_sampling_updates(self, n_steps=1, use_pcd=True):
        """
        Implements the negative phase, generating samples from p(h,s,v).
        :param n_steps: scalar, number of Gibbs steps to perform.
        """
        init_chain = self.neg_v if use_pcd else self.input
        [new_g, new_h, new_s, new_v] =  self.neg_sampling(
                self.neg_g, self.neg_h, self.neg_v,
                n_steps = n_steps)

        updates = OrderedDict()
        updates[self.neg_g] = new_g
        updates[self.neg_h] = new_h
        updates[self.neg_s] = new_s
        updates[self.neg_v] = new_v

        return updates

    def ml_cost(self, pos_g, pos_h, pos_v, neg_g, neg_h, neg_v, mean_field=False):
        """
        Variational approximation to the maximum likelihood positive phase.
        :param v: T.matrix of shape (batch_size, n_v), training examples
        :return: tuple (cost, gradient)
        """
        pos_cost, pos_cte = self.free_energy(pos_g, pos_h, pos_v)
        neg_cost, neg_cte = self.free_energy(neg_g, neg_h, neg_v)
        cost = pos_cost - neg_cost

        # build gradient of cost with respect to model parameters
        cte = pos_cte + neg_cte

        return costmod.Cost(cost, self.params(), cte)


    ##############################
    # GENERIC OPTIMIZATION STUFF #
    ##############################
    def get_sparsity_cost(self, pos_g, pos_h):

        # update mean activation using exponential moving average
        hack_g = self.g_given_hv(pos_h, self.input)
        hack_h = self.h_given_gv(pos_g, self.input)

        # define loss based on value of sp_type
        eps = npy_floatX(1./self.batch_size)
        loss = lambda targ, val: - npy_floatX(targ) * T.log(eps + val) \
                                 - npy_floatX(1-targ) * T.log(1 - val + eps)

        params = []
        cost = T.zeros((), dtype=floatX)
        if self.sp_weight['g'] or self.sp_weight['h']:
            params += [self.Wv, self.alpha, self.mu]
            if self.sp_weight['g']:
                cost += self.sp_weight['g']  * T.sum(loss(self.sp_targ['g'], hack_g.mean(axis=0)))
                params += [self.gbias]
            if self.sp_weight['h']:
                cost += self.sp_weight['h']  * T.sum(loss(self.sp_targ['h'], hack_h.mean(axis=0)))
                params += [self.hbias]
            if self.flags['split_norm']:
                params += [self.scalar_norms]
        cte = [pos_g, pos_h]
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
        if self.flags['split_norm']:
            chans.update(self.monitor_vector(self.scalar_norms))
        chans.update(self.monitor_matrix(self.Wv))
        chans.update(self.monitor_matrix(self.Wg))
        chans.update(self.monitor_matrix(self.Wh))
        chans.update(self.monitor_vector(self.gbias))
        chans.update(self.monitor_vector(self.hbias))
        chans.update(self.monitor_vector(self.vbias))
        chans.update(self.monitor_vector(self.alpha_prec, name='alpha_prec'))
        chans.update(self.monitor_vector(self.mu))
        chans.update(self.monitor_matrix(self.neg_g))
        chans.update(self.monitor_matrix(self.neg_h))
        chans.update(self.monitor_matrix(self.neg_s - self.mu, name='(neg_s - mu)'))
        chans.update(self.monitor_matrix(self.neg_v))
        wg_norm = T.sqrt(T.sum(self.Wg**2, axis=0))
        wh_norm = T.sqrt(T.sum(self.Wh**2, axis=0))
        wv_norm = T.sqrt(T.sum(self.Wv**2, axis=0))
        chans.update(self.monitor_vector(wg_norm, name='wg_norm'))
        chans.update(self.monitor_vector(wh_norm, name='wh_norm'))
        chans.update(self.monitor_vector(wv_norm, name='wv_norm'))
        chans['lr'] = self.lr
        return chans


class TrainingAlgorithm(default.DefaultTrainingAlgorithm):

    def setup(self, model, dataset):
        x = dataset.get_batch_design(10000, include_labels=False)
        ml_vbias = rbm_utils.compute_ml_bias(x)
        model.vbias.set_value(ml_vbias)
        neg_v = model.rng.binomial(n=1, p=model.vbias.get_value(), size=(model.batch_size, model.n_v))
        super(TrainingAlgorithm, self).setup(model, dataset)
