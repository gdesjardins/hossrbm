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

class BilinearSpikeSlabRBM(Model, Block):
    """Spike & Slab Restricted Boltzmann Machine (RBM)  """

    def validate_flags(self, flags):
        flags.setdefault('truncate_s', False)
        flags.setdefault('truncate_v', False)
        flags.setdefault('scalar_lambd', False)
        flags.setdefault('wg_norm', 'none')
        flags.setdefault('wh_norm', 'none')
        flags.setdefault('wv_norm', 'none')
        flags.setdefault('split_norm', False)
        flags.setdefault('ml_lambd', False)
        flags.setdefault('init_mf_rand', False)
        flags.setdefault('center_g', False)
        flags.setdefault('center_h', False)
        flags.setdefault('igo_init', False)
        if len(flags.keys()) != 13:
            raise NotImplementedError('One or more flags are currently not implemented.')

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

        # init scalar norm for each entry of Wv
        sn_val = self.iscales['scalar_norms'] * numpy.ones(self.n_s)
        self.scalar_norms = sharedX(sn_val, name='scalar_norms')

        if self.flags['igo_init']:
            print 'Overriding iscales initialization with 1./sqrt(nv x nh)'
            self.iscales['Wv'] = 1./numpy.sqrt(max(self.n_v, self.n_s))
            self.iscales['Wg'] = 1./numpy.sqrt(max(self.n_g, self.n_s))
            self.iscales['Wh'] = 1./numpy.sqrt(max(self.n_h, self.n_s))

        # Init (visible, slabs) weight matrix.
        self.Wv = self.init_weight(self.iscales['Wv'], (self.n_v, self.n_s), 'Wv',
                        normalize= (self.flags['wv_norm'] == 'unit'))
        self.norm_wv = T.sqrt(T.sum(self.Wv**2, axis=0))

        # Initialize (slab, hidden) pooling matrix
        self.Wh = sharedX(self.sparse_hmask.mask.T * self.iscales.get('Wh', 1.0), name='Wh')

        # Initialize (slabs, g-unit) weight matrix.
        if self.sparse_gmask:
            self.Wg = sharedX(self.sparse_gmask.mask.T * self.iscales.get('Wg', 1.0), name='Wg')
        else:
            self.Wg = self.init_weight(self.iscales['Wg'], (self.n_s, self.n_g), 'Wg')

        # allocate shared variables for bias parameters
        self.gbias = sharedX(self.iscales['gbias'] * numpy.ones(self.n_g), name='gbias')
        self.hbias = sharedX(self.iscales['hbias'] * numpy.ones(self.n_h), name='hbias')
        self.cg = sharedX(0.5 * numpy.ones(self.n_g), name='cg')
        self.ch = sharedX(0.5 * numpy.ones(self.n_h), name='ch')

        # mean (mu) and precision (alpha) parameters on s
        self.mu = self.norm_wv
        self.alpha = sharedX(self.iscales['alpha'] * numpy.ones(self.n_s), name='alpha')
        self.alpha_prec = T.nnet.softplus(self.alpha)

        # diagonal of precision matrix of visible units
        self.lambd = sharedX(self.iscales['lambd'] * numpy.ones(self.n_v), name='lambd')
        self.lambd_prec = T.nnet.softplus(self.lambd)

    def init_chains(self):
        """ Allocate shared variable for persistent chain """
        # initialize buffers to store inference state
        self.pos_g  = sharedX(numpy.zeros((self.batch_size, self.n_g)), name='pos_g')
        self.pos_h  = sharedX(numpy.zeros((self.batch_size, self.n_h)), name='pos_h')
        self.pos_s1 = sharedX(numpy.zeros((self.batch_size, self.n_s)), name='pos_s1')
        self.pos_s0 = sharedX(numpy.zeros((self.batch_size, self.n_s)), name='pos_s0')
        # initialize visible unit chains
        scale = numpy.sqrt(1./softplus(self.lambd.get_value()))
        neg_v  = self.rng.normal(loc=0, scale=scale, size=(self.batch_size, self.n_v))
        self.neg_v  = sharedX(neg_v, name='neg_v')
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
        self.pos_counter  = sharedX(0., name='pos_counter')
        self.odd_even = sharedX(0., name='odd_even')
 
    def params(self):
        """
        Returns a list of learnt model parameters.
        """
        params = [self.Wv, self.Wg, self.hbias, self.gbias, self.alpha, self.lambd]
        if self.flags['split_norm']:
            params += [self.scalar_norms]
        return params

    def do_theano(self):
        """ Compiles all theano functions needed to use the model"""

        init_names = dir(self)

        ###### All fields you don't want to get pickled (e.g., theano functions) should be created below this line
        self.init_debug()

        # SAMPLING: NEGATIVE PHASE
        neg_updates = self.neg_sampling_updates(n_steps=self.neg_sample_steps, use_pcd=True)
        self.sample_func = theano.function([], [], updates=neg_updates)

        # POSITIVE PHASE
        pos_updates = self.pos_phase_updates(
                self.input,
                n_steps = self.pos_steps)

        pos_g = pos_updates[self.pos_g]
        pos_h = pos_updates[self.pos_h]
        pos_s1 = pos_updates[self.pos_s1]
        pos_s0 = pos_updates[self.pos_s0]
        pos_s = self.s_hat(pos_h, pos_s1, pos_s0)
        self.inference_func = theano.function([self.input], [pos_g, pos_h, pos_s])

        ##
        # BUILD COST OBJECTS
        ##
        lcost = self.ml_cost(
                        pos_g = pos_g,
                        pos_h = pos_h,
                        pos_s1 = pos_s1,
                        pos_s0 = pos_s0,
                        pos_v = self.input,
                        neg_g = neg_updates[self.neg_g],
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

        ##
        # BUILD UPDATES DICTIONARY FROM GRADIENTS
        ##
        learning_updates = costmod.get_updates(learning_grads)
        learning_updates.update(pos_updates)
        learning_updates.update(neg_updates)
        learning_updates.update({self.iter: self.iter+1})

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

    def get_constraint_updates(self):
        constraint_updates = OrderedDict() 

        if self.flags['scalar_lambd']:
            constraint_updates[self.lambd] = T.mean(self.lambd) * T.ones_like(self.lambd)

        ## Enforce sparsity pattern on g if required ##
        if self.sparse_gmask:
            constraint_updates[self.Wg] = self.Wg * self.sparse_gmask.mask.T

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
        if self.flags['truncate_v']:
            x = numpy.clip(x, -self.truncation_bound['v'], self.truncation_bound['v'])
        try:
            self.batch_train_func(x)
            self.enforce_constraints()
        except:
            import pdb; pdb.set_trace()

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

    def energy(self, g_sample, h_sample, s_sample, v_sample):
        from_v = self.from_v(v_sample)
        from_h = self.from_h(h_sample)
        from_g = self.from_g(g_sample)
        cg_sample = g_sample - self.cg if self.flags['center_g'] else g_sample
        ch_sample = h_sample - self.ch if self.flags['center_h'] else h_sample

        energy  = 0.
        energy -= T.sum(from_v * self.alpha_prec * s_sample * from_h, axis=1)
        energy += 0.5 * T.sum(self.alpha_prec * s_sample**2, axis=1)
        energy += T.sum(0.5 * self.lambd_prec * v_sample**2, axis=1)
        energy -= T.sum(self.alpha_prec * self.mu * s_sample, axis=1)
        energy -= T.sum(self.alpha_prec * (s_sample - self.mu) * from_g, axis=1)
        energy -= T.dot(cg_sample, self.gbias)
        energy -= T.dot(ch_sample, self.hbias)
        return energy, [g_sample, h_sample, s_sample, v_sample]

    def eq_log_pstar_vgh(self, g_hat, h_hat, s1_hat, s0_hat, v):
        """
        Computes the expectation (under the variational distribution q(g,h)=q(g)q(h)) of the
        log un-normalized probability, i.e. log p^*(g,h,s,v)
        :param g_hat: T.matrix of shape (batch_size, n_g)
        :param h_hat: T.matrix of shape (batch_size, n_h)
        :param v    : T.matrix of shape (batch_size, n_v)
        """
        from_v = self.from_v(v)
        from_h = self.from_h(h_hat)
        from_g = self.from_g(g_hat)

        # center variables
        cg_hat = g_hat - self.cg if self.flags['center_g'] else g_hat
        ch_hat = h_hat - self.ch if self.flags['center_h'] else h_hat
        # compute expectation of various s-quantities
        s_hat  = self.s_hat(ch_hat, s1_hat, s0_hat)
        ss_hat = self.s_hat(ch_hat, s1_hat**2 + 1./self.alpha_prec,
                                    s0_hat**2 + 1./self.alpha_prec)

        lq  = 0.
        lq += T.sum(from_v * self.alpha_prec * s1_hat * from_h, axis=1)
        lq -= 0.5 * T.sum(self.alpha_prec * ss_hat, axis=1)
        lq -= T.sum(0.5 * self.lambd_prec * v**2, axis=1)
        lq += T.sum(self.alpha_prec * self.mu * s_hat, axis=1)
        lq += T.sum(self.alpha_prec * from_g  * (s_hat - self.mu), axis=1)
        lq += T.dot(cg_hat, self.gbias)
        lq += T.dot(ch_hat, self.hbias)
        return T.mean(lq), [g_hat, h_hat, s_hat, ss_hat, s1_hat, s0_hat, v]

    def __call__(self, v, output_type='g+h'):
        print 'Building representation with %s' % output_type
        init_state = OrderedDict()
        init_state['g'] = T.ones((v.shape[0],self.n_g)) * T.nnet.sigmoid(self.gbias)
        init_state['h'] = T.ones((v.shape[0],self.n_h)) * T.nnet.sigmoid(self.hbias)
        [g, h, s2_1, s2_0, v, pos_counter] = self.pos_phase(v, init_state, n_steps=self.pos_steps)
        s = s2_1

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
        Wv = self.Wv / self.norm_wv
        if self.flags['split_norm']:
            Wv *= self.scalar_norms
        return T.dot(self.lambd_prec * v_sample, Wv)

    def from_g(self, g_sample):
        if self.flags['center_g']:
            g_sample = g_sample - self.cg
        return T.dot(g_sample, self.Wg.T)

    def from_h(self, h_sample):
        if self.flags['center_h']:
            h_sample = h_sample - self.ch
        return T.dot(h_sample, self.Wh.T)

    def to_g(self, g_s):
        return T.dot(g_s, self.Wg)

    def to_h(self, h_s):
        return T.dot(h_s, self.Wh)

    def g_given_s(self, s_sample):
        g_mean_s = self.alpha_prec * (s_sample - self.mu)
        g_mean = self.to_g(g_mean_s) + self.gbias
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

    def h_given_gv(self, g_sample, v_sample):
        from_v = self.from_v(v_sample)
        from_g = self.from_g(g_sample)
        h_mean_s = 0.5 * self.alpha_prec * from_v**2
        h_mean_s += self.alpha_prec * from_v * (self.mu + from_g)
        h_mean = self.to_h(h_mean_s) + self.hbias
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

    def s_given_ghv(self, g_sample, h_sample, v_sample):
        from_g = self.from_g(g_sample)
        from_h = self.from_h(h_sample)
        from_v = self.from_v(v_sample)
        s_mean = self.mu + from_g +  from_v * from_h
        return s_mean

    def sample_s_given_ghv(self, g_sample, h_sample, v_sample, rng=None, size=None):
        """
        Generates sample from p(s | g, h, v)
        """
        s_mean = self.s_given_ghv(g_sample, h_sample, v_sample)
        
        rng = self.theano_rng if rng is None else rng
        size = size if size else self.batch_size

        if self.flags['truncate_s']:
            s_sample = truncated.truncated_normal(
                    size=(size, self.n_s),
                    avg = s_mean, 
                    std = T.sqrt(1./self.alpha_prec),
                    lbound = self.mu - self.truncation_bound['s'],
                    ubound = self.mu + self.truncation_bound['s'],
                    theano_rng = rng,
                    dtype=floatX)
        else: 
            s_sample = rng.normal(
                    size=(size, self.n_s),
                    avg = s_mean, 
                    std = T.sqrt(1./self.alpha_prec),
                    dtype=floatX)
        return s_sample

    def v_given_hs(self, h_sample, s_sample):
        Wv = self.Wv
        if self.flags['split_norm']:
            Wv *= self.scalar_norms

        from_h = self.from_h(h_sample)
        v_mean = T.dot(self.alpha_prec * s_sample * from_h, Wv.T)
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

    ########################################
    # FIXED POINT EQUATIONS FOR MEAN-FIELD #
    ########################################
    def g_hat(self, h_hat, s1_hat, s0_hat):
        s_hat  = self.s_hat(h_hat, s1_hat, s0_hat)
        g_hat_s = self.alpha_prec * (s_hat - self.mu)
        g_hat_mean = self.to_g(g_hat_s) + self.gbias
        return T.nnet.sigmoid(g_hat_mean)

    def h_hat(self, g_hat, v):
        return self.h_given_gv(g_hat, v)

    def s_hat(self, h_hat, s1_hat, s0_hat):
        """ 
        s_hat := E_{q(s|h)q(h)}[s_ij]
               = E[s_ij | h_j=1] p(h_j = 1) + E[s_ij | h_j=0] p(h_j = 0)
        """
        from_h = self.from_h(h_hat)
        return s1_hat * from_h + s0_hat * (1 - from_h)

    def s1_hat(self, g_hat, v):
        from_v = self.from_v(v)
        from_g = self.from_g(g_hat)
        s1_hat = from_v + from_g + self.mu
        return s1_hat

    def s0_hat(self, g_hat, v):
        from_g = self.from_g(g_hat)
        s0_hat = from_g + self.mu
        return s0_hat

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
        def pos_mf_iteration(g1, h1, v, pos_counter):
            h2 = self.h_hat(g1, v)
            s2_1 = self.s1_hat(g1, v)
            s2_0 = self.s0_hat(g1, v)
            g2 = self.g_hat(h2, s2_1, s2_0)
            # stopping criterion
            stop = T.maximum(T.max(g2 - g1), T.max(h2 - h1))
            return [g2, h2, s2_1, s2_0, v, pos_counter + 1], theano.scan_module.until(stop < eps)

        states = [T.unbroadcast(T.shape_padleft(init_state['g'])),
                  T.unbroadcast(T.shape_padleft(init_state['h'])),
                  {'steps': 1}, {'steps': 1},
                  T.unbroadcast(T.shape_padleft(v)),
                  T.unbroadcast(T.shape_padleft(0.))]

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
            init_state['h'] = T.ones((v.shape[0], self.n_h)) * T.nnet.sigmoid(self.hbias)

        [new_g, new_h, new_s1, new_s0, crap_v, pos_counter] = self.pos_phase(
                v, init_state=init_state, n_steps=n_steps)

        # update running average of positive phase activations
        pos_updates = OrderedDict()
        pos_updates[self.pos_counter] = pos_counter
        pos_updates[self.odd_even] = (self.odd_even + 1) % 2
        pos_updates[self.pos_g] = new_g
        pos_updates[self.pos_h] = new_h
        pos_updates[self.pos_s1] = new_s1
        pos_updates[self.pos_s0] = new_s0
        return pos_updates

    def neg_sampling(self, g_sample, h_sample, s_sample, v_sample, n_steps=1):
        """
        Gibbs step for negative phase, which alternates:
        p(g|b,h,v), p(h|b,g,v), p(b|g,h,v), p(s|b,g,h,v) and p(v|b,g,h,s)
        :param g_sample: T.matrix of shape (batch_size, n_g)
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        :param n_steps: number of Gibbs updates to perform in negative phase.
        """

        def gibbs_iteration(g1, h1, s1, v1):
            g2 = self.sample_g_given_s(s1)
            h2 = self.sample_h_given_gv(g2, v1)
            s2 = self.sample_s_given_ghv(g2, h2, v1)
            v2 = self.sample_v_given_hs(h2, s2)
            return [g2, h2, s2, v2]

        rvals , updates = theano.scan(
                gibbs_iteration,
                outputs_info = [g_sample, h_sample, s_sample, v_sample],
                n_steps=n_steps)
        
        return [rval[-1] for rval in rvals]

    def neg_sampling_updates(self, n_steps=1, use_pcd=True):
        """
        Implements the negative phase, generating samples from p(h,s,v).
        :param n_steps: scalar, number of Gibbs steps to perform.
        """
        init_chain = self.neg_v if use_pcd else self.input
        [new_g, new_h, new_s, new_v] =  self.neg_sampling(
                self.neg_g, self.neg_h, self.neg_s, self.neg_v,
                n_steps = n_steps)

        updates = OrderedDict()
        updates[self.neg_g] = new_g
        updates[self.neg_h] = new_h
        updates[self.neg_s] = new_s
        updates[self.neg_v] = new_v
        return updates

    def ml_cost(self, pos_g, pos_h, pos_s1, pos_s0, pos_v,
                      neg_g, neg_h, neg_s, neg_v):
        """
        Variational approximation to the maximum likelihood positive phase.
        :param v: T.matrix of shape (batch_size, n_v), training examples
        :return: tuple (cost, gradient)
        """
        # L(q) = pos_cost + neg_cost + H(q)
        # pos_cost := E_{q(g)q(h)} log p(g,h,v)
        # neg_cost := -log Z
        pos_cost, pos_cte = self.eq_log_pstar_vgh(pos_g, pos_h, pos_s1, pos_s0, pos_v)
        # - dlogZ/dtheta = E_p[ denergy / dtheta ]
        neg_cost, neg_cte = self.energy(neg_g, neg_h, neg_s, neg_v)
        # build gradient of cost with respect to model parameters
        cost = - (pos_cost + T.mean(neg_cost))
        cte = pos_cte + neg_cte + [self.norm_wv]
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

    def monitor_vector(self, b, name=None):
        if name is None: assert hasattr(b, 'name')
        name = name if name else b.name

        rval = OrderedDict()
        rval[name + '.min'] = b.min()
        rval[name + '.max'] = b.max()
        rval[name + '.absmean'] = abs(b).mean()
        return rval

    def get_monitoring_channels(self, x, y=None):
        chans = OrderedDict()
        if self.flags['split_norm']:
            chans.update(self.monitor_vector(self.scalar_norms))
        chans.update(self.monitor_matrix(self.Wv))
        chans.update(self.monitor_matrix(self.Wg))
        chans.update(self.monitor_matrix(self.Wh))
        chans.update(self.monitor_vector(self.gbias))
        chans.update(self.monitor_vector(self.hbias))
        chans.update(self.monitor_vector(self.alpha_prec, name='alpha_prec'))
        chans.update(self.monitor_vector(self.mu, name='mu'))
        chans.update(self.monitor_vector(self.lambd_prec, name='lambd_prec'))
        chans.update(self.monitor_matrix(self.pos_g))
        chans.update(self.monitor_matrix(self.pos_h))
        chans.update(self.monitor_matrix(self.pos_s1 - self.mu, name='(pos_s1 - mu)', abs_mean=False))
        chans.update(self.monitor_matrix(self.pos_s0 - self.mu, name='(pos_s0 - mu)', abs_mean=False))
        chans.update(self.monitor_matrix(self.neg_g))
        chans.update(self.monitor_matrix(self.neg_h))
        chans.update(self.monitor_matrix(self.neg_s - self.mu, name='(neg_s - mu)'))
        chans.update(self.monitor_matrix(self.neg_v))
        wg_norm = T.sqrt(T.sum(self.Wg**2, axis=0))
        wv_norm = T.sqrt(T.sum(self.Wv**2, axis=0))
        chans.update(self.monitor_vector(wg_norm, name='wg_norm'))
        chans.update(self.monitor_vector(wv_norm, name='wv_norm'))
        chans['lr'] = self.lr
        chans['pos_counter'] = self.pos_counter
        return chans

    def init_debug(self):
        neg_g = self.g_given_s(self.neg_s)
        neg_h = self.h_given_gv(self.neg_g, self.neg_v)
        neg_s = self.s_given_ghv(self.neg_g, self.neg_h, self.neg_v)
        neg_v = self.v_given_hs(self.neg_h, self.neg_s)
        self.sample_g_func = theano.function([], neg_g)
        self.sample_h_func = theano.function([], neg_h)
        self.sample_s_func = theano.function([], neg_s)
        self.sample_v_func = theano.function([], neg_v)

        # Build function to compute energies.
        gg = T.matrix('g')
        hh = T.matrix('h')
        ss = T.matrix('s')
        vv = T.matrix('v')
        E, _crap = self.energy(gg,hh,ss,vv)
        self.energy_func = theano.function([gg,hh,ss,vv], E)


import pylab as pl

class TrainingAlgorithm(default.DefaultTrainingAlgorithm):

    def init_params_from_data(self, model, x):

        if model.flags['ml_lambd']:
            # compute maximum likelihood solution for lambd
            scale = (1./numpy.std(x, axis=0))**2
            model.lambd.set_value(softplus_inv(scale).astype(floatX))
            # reset neg_v markov chain accordingly
            neg_v = model.rng.normal(loc=0, scale=scale, size=(model.batch_size, model.n_v))
            model.neg_v.set_value(neg_v.astype(floatX))

        """
        [pos_g, pos_h, pos_s] = model.inference_func(x)
        if model.flags['center_g']:
            model.cg.set_value(pos_g.mean(axis=0).astype(floatX))
        if model.flags['center_h']:
            model.ch.set_value(pos_h.mean(axis=0).astype(floatX))
  
        [pos_g, pos_h, pos_s] = model.inference_func(x)
        e1 = model.energy_func(pos_g, pos_h, pos_s, x)
        pl.hist(e1, bins=100); pl.show()

        if model.flags['igo_init']:
            hbias_init = rbm_utils.compute_ml_bias(pos_h)
            model.hbias.set_value(-hbias_init)
            gbias_init = rbm_utils.compute_ml_bias(pos_g)
            model.gbias.set_value(-gbias_init)
        
        [pos_g, pos_h, pos_s] = model.inference_func(x); e1 = model.energy_func(pos_g, pos_h, pos_s, x); pl.hist(e1, bins=100); pl.show()

        import pdb; pdb.set_trace()
        """


    def setup(self, model, dataset):

        x = dataset.get_batch_design(10000, include_labels=False)
        self.init_params_from_data(model, x)
        super(TrainingAlgorithm, self).setup(model, dataset)
