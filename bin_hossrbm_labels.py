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

from pylearn2.training_algorithms import default
from pylearn2.utils import serial
from pylearn2.base import Block
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace

import truncated
import cost as costmod
from utils import tools
from utils import sharedX, floatX, npy_floatX
from utils import rbm_utils
from true_gradient import true_gradient

def sigm(x): return 1./(1 + numpy.exp(-x))
def softplus(x): return numpy.log(1. + numpy.exp(x))
def softplus_inv(x): return numpy.log(numpy.exp(x) - 1.)
def softmax(x):
    assert x.ndim == 1
    max_x = numpy.max(x)
    return numpy.exp(x - max_x) / numpy.sum(numpy.exp(x - max_x))

class BinaryBilinearSpikeSlabRBMWithLabels(Model, Block):
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
        flags.setdefault('truncate_s', False)
        flags.setdefault('truncate_v', False)
        flags.setdefault('scalar_lambd', False)
        flags.setdefault('lambd_interaction', False)
        flags.setdefault('wg_norm', 'none')
        flags.setdefault('wh_norm', 'none')
        flags.setdefault('wv_norm', 'none')
        flags.setdefault('split_norm', False)
        flags.setdefault('mean_field', False)
        flags.setdefault('ml_lambd', False)
        if len(flags.keys()) != 10:
            raise NotImplementedError('One or more flags are currently not implemented.')

    def __init__(self, 
            numpy_rng = None, theano_rng = None,
            n_g=99, n_h=99, n_l=10, n_s=99, n_v=100, label_multiplier=10., init_from=None,
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
        self.input_labels = T.matrix('input_labels')
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

    def init_weight(self, iscale, shape, name, normalize=True, axis=0):
        value =  self.rng.normal(size=shape) * iscale
        if normalize:
            value /= numpy.sqrt(numpy.sum(value**2, axis=axis))
        return sharedX(value, name=name)

    def init_parameters(self):
        # init scalar norm for each entry of Wv
        sn_val = self.iscales['scalar_norms'] * numpy.ones(self.n_s)
        self.scalar_norms = sharedX(sn_val, name='scalar_norms')

        # init weight matrices
        normalize_wv = self.flags['wv_norm'] == 'unit' 
        self.Wv = self.init_weight(self.iscales['Wv'], (self.n_v, self.n_s), 'Wv', normalize=normalize_wv)
        self.Whl = self.init_weight(self.iscales['Whl'], (self.n_h, self.n_l), 'Whl', normalize=False)
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
        self.lbias = sharedX(self.iscales['lbias'] * numpy.ones(self.n_l), name='lbias')
        self.vbias = sharedX(self.iscales['vbias'] * numpy.ones(self.n_v), name='vbias')

        # mean (mu) and precision (alpha) parameters on s
        self.mu = sharedX(self.iscales['mu'] * numpy.ones(self.n_s), name='mu')
        self.alpha = sharedX(self.iscales['alpha'] * numpy.ones(self.n_s), name='alpha')
        self.alpha_prec = T.nnet.softplus(self.alpha)

    def init_chains(self):
        """ Allocate shared variable for persistent chain """
        # initialize s-chain
        loc = self.mu.get_value()
        scale = numpy.sqrt(1./softplus(self.alpha.get_value()))
        neg_s  = self.rng.normal(loc=loc, scale=scale, size=(self.batch_size, self.n_s))
        self.neg_s  = sharedX(neg_s, name='neg_s')
        # initialize binary g-h-v chains
        pval_g = sigm(self.gbias.get_value())
        pval_h = sigm(self.hbias.get_value())
        pval_l = softmax(self.lbias.get_value())
        neg_g = self.rng.binomial(n=1, p=pval_g, size=(self.batch_size, self.n_g))
        neg_h = self.rng.binomial(n=1, p=pval_h, size=(self.batch_size, self.n_h))
        neg_v = self.rng.binomial(n=1, p=pval_v, size=(self.batch_size, self.n_v))
        neg_l = self.rng.multinomial(n=1, pvals=pval_l, size=(self.batch_size))
        self.neg_h  = sharedX(neg_h, name='neg_h')
        self.neg_g  = sharedX(neg_g, name='neg_g')
        self.neg_v  = sharedX(neg_v, name='neg_v')
        self.neg_l  = sharedX(neg_l, name='neg_l')
        # other misc.
        self.pos_counter  = sharedX(0., name='pos_counter')
        self.odd_even = sharedX(0., name='odd_even')
 
    def params(self):
        """
        Returns a list of learnt model parameters.
        """
        params = [self.Wv, self.Whl, self.hbias, self.gbias, self.vbias, self.lbias, self.alpha, self.mu]
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

        ### BUILD TWO FUNCTIONS: ONE CONDITIONED ON LABELS, ONE WHERE LABELS ARE SAMPLED
        self.batch_train_func = {}
        for (key, input_label) in zip(['nolabel','label'], [None, self.input_labels]):

            # POSITIVE PHASE
            pos_states, pos_updates = self.pos_phase_updates(
                    self.input, l = input_label,
                    n_steps = self.pos_steps,
                    mean_field=self.flags['mean_field'])

            ##
            # BUILD COST OBJECTS
            ##
            lcost = self.ml_cost(
                            pos_g = pos_states['g'],
                            pos_h = pos_states['h'],
                            pos_l = pos_states['l'],
                            pos_v = self.input,
                            neg_g = neg_updates[self.neg_g],
                            neg_h = neg_updates[self.neg_h],
                            neg_v = neg_updates[self.neg_v],
                            neg_l = neg_updates[self.neg_l],
                            mean_field=self.flags['mean_field'])
            spcost = self.get_sparsity_cost(pos_states['g'], pos_states['h'], pos_states['l'])
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
            inputs = [self.input, self.input_labels] if key == 'label' else [self.input]
            self.batch_train_func[key] = function(inputs, [],
                                             updates=learning_updates,
                                             name='train_rbm_func_%s' % key)

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

        (x, y) = dataset.get_batch_design(batch_size, include_labels=True)
        if self.flags['truncate_v']:
            x = numpy.clip(x, -self.truncation_bound['v'], self.truncation_bound['v'])
        if y is None:
            self.batch_train_func['nolabel'](x)
        else:
            self.batch_train_func['label'](x, y)
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

    def free_energy(self, g_sample, h_sample, v_sample, l_sample):
        """
        Computes energy for a given configuration of (g,h,v,x,y).
        :param h_sample: T.matrix of shape (batch_size, n_g)
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        :param l_sample: T.matrix of shape (batch_size, n_l)
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
        energy += self.label_energy(h_sample, l_sample)
        return T.mean(energy), [g_sample, h_sample, v_sample, l_sample]

    def label_energy(self, h_sample, l_sample):
        energy = -T.sum(l_sample * T.dot(h_sample, self.Whl), axis=1)
        energy -= T.dot(l_sample, self.lbias)
        return self.label_multiplier * energy

    def __call__(self, v, output_type='g+h', mean_field=True):
        print 'Building representation with %s' % output_type
        init_state = OrderedDict()
        init_state['g'] = T.ones((v.shape[0],self.n_g)) * T.nnet.sigmoid(self.gbias)
        init_state['h'] = T.ones((v.shape[0],self.n_h)) * T.nnet.sigmoid(self.hbias)
        init_state['l'] = T.ones((v.shape[0],self.n_l)) * T.nnet.softmax(self.lbias)
        [g, h, l] = self.pos_phase(v, init_state, n_steps=self.pos_steps, mean_field=mean_field)
        s = self.s_given_ghv(g, h, v)

        atoms = {
            'g_s' : T.dot(g, self.Wg),  # g in s-space
            'h_s' : T.dot(h, self.Wh),  # h in s-space
            's_g' : T.sqrt(T.dot(s**2, self.Wg.T)),
            's_h' : T.sqrt(T.dot(s**2, self.Wh.T)),
            's_g__h' : T.sqrt(T.dot(s**2 * T.dot(h, self.Wh), self.Wg.T)),
            's_h__g' : T.sqrt(T.dot(s**2 * T.dot(g, self.Wg), self.Wh.T))
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
        Wv = self.Wv
        if self.flags['split_norm']:
            Wv *= self.scalar_norms
        return T.dot(v_sample, Wv)

    def from_g(self, g_sample):
        return T.dot(g_sample, self.Wg)

    def from_h(self, h_sample):
        return T.dot(h_sample, self.Wh)

    def to_g(self, g_s):
        return T.dot(g_s, self.Wg.T)

    def to_h(self, h_s):
        return T.dot(h_s, self.Wh.T)

    def l_given_h(self, h_sample):
        """
        Compute p(l_i = 1 | h)
        :param h_sample: T.matrix of shape (batch_size, n_l matrix)
        """
        l_mean = self.label_multiplier * (T.dot(h_sample, self.Whl) + self.lbias)
        return T.nnet.softmax(l_mean)
 
    def sample_l_given_h(self, h_sample, rng=None, size=None):
        """
        Generates sample from p(l | h)
        """
        l_mean = self.l_given_h(h_sample)
        rng = self.theano_rng if rng is None else rng
        size = size if size else self.batch_size
        l_sample = rng.multinomial(size=(size,self.n_l), n=1,
            pvals=l_mean, dtype=floatX)
        return l_sample
 
    def h_given_gvl_input(self, g_sample, v_sample, l_sample):
        """
        Compute mean activation of h given v.
        :param g_sample: T.matrix of shape (batch_size, n_g matrix)
        :param v_sample: T.matrix of shape (batch_size, n_v matrix)
        """
        from_v = self.from_v(v_sample)
        from_g = self.from_g(g_sample)
        h_mean_s = 0.5 * 1./self.alpha_prec * from_g * from_v**2
        h_mean_s += from_g * from_v * self.mu
        h_mean = self.to_h(h_mean_s) + self.hbias
        h_mean += self.label_multiplier * T.dot(l_sample, self.Whl.T)
        return h_mean
    
    def h_given_gvl(self, g_sample, v_sample, l_sample):
        h_mean = self.h_given_gvl_input(g_sample, v_sample, l_sample)
        return T.nnet.sigmoid(h_mean)

    def sample_h_given_gvl(self, g_sample, v_sample, l_sample, rng=None, size=None):
        """
        Generates sample from p(h | g, v)
        """
        h_mean = self.h_given_gvl(g_sample, v_sample, l_sample)

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
        g_mean = self.to_g(g_mean_s) + self.gbias
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
        return T.nnet.sigmoid(v_mean)

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
    def pos_phase(self, v, init_state, l=None, n_steps=1, mean_field=False, eps=1e-3):
        """
        Mixed mean-field + sampling inference in positive phase.
        :param v: input being conditioned on
        :param init: dictionary of initial values
        :param n_steps: number of Gibbs updates to perform afterwards.
        """
        assert mean_field

        def pos_mf_iteration(g1, h1, l1, pos_counter, v, size):
            # branch1: start with g
            branch1_g2 = self.g_given_hv(h1, v)
            branch1_h2 = self.h_given_gvl(branch1_g2, v, l1)
            branch1_l2 = self.l_given_h(branch1_h2)
            # branch2: start with h
            branch2_h2 = self.h_given_gvl(g1, v, l1)
            branch2_l2 = self.l_given_h(branch2_h2)
            branch2_g2 = self.g_given_hv(branch2_h2, v)
            # decide which way we should sample
            g2 = ifelse(self.odd_even, branch1_g2, branch2_g2)
            h2 = ifelse(self.odd_even, branch1_h2, branch2_h2)
            l2 = ifelse(self.odd_even, branch1_l2, branch2_l2)
            # stopping criterion
            dl_dghat = T.max(abs(self.dfe_dghat(g2, h2, l2, v)))
            dl_dhhat = T.max(abs(self.dfe_dhhat(g2, h2, l2, v)))
            dl_dlhat = T.max(abs(self.dfe_dlhat(g2, h2, l2, v)))
            stop = T.max((dl_dghat, dl_dhhat, dl_dlhat))
            return [g2, h2, l2, pos_counter + 1], theano.scan_module.until(stop < eps)

        def pos_mf_iteration_labels(g1, h1, pos_counter, v, l, size):
            # branch1: start with g
            branch1_g2 = self.g_given_hv(h1, v)
            branch1_h2 = self.h_given_gvl(branch1_g2, v, l)
            # branch2: start with h
            branch2_h2 = self.h_given_gvl(g1, v, l)
            branch2_g2 = self.g_given_hv(branch2_h2, v)
            # decide which way we should sample
            g2 = ifelse(self.odd_even, branch1_g2, branch2_g2)
            h2 = ifelse(self.odd_even, branch1_h2, branch2_h2)
            # stopping criterion
            dl_dghat = T.max(abs(self.dfe_dghat(g2, h2, l, v)))
            dl_dhhat = T.max(abs(self.dfe_dhhat(g2, h2, l, v)))
            stop = T.max((dl_dghat, dl_dhhat))
            return [g2, h2, pos_counter + 1], theano.scan_module.until(stop < eps)

        iter_func = pos_mf_iteration if l is None else pos_mf_iteration_labels
        non_sequences = [v, v.shape[0]] if l is None else [v, l, v.shape[0]]

        # define initial conditions for loop
        outputs_info = [init_state['g'], init_state['h']]
        if l is None:
            outputs_info += [init_state['l']]
        outputs_info += [0.]

        outputs, updates = theano.scan(
                iter_func,
                outputs_info = outputs_info,
                non_sequences = non_sequences,
                n_steps=n_steps)

        return [output[-1] for output in outputs]

    def pos_phase_updates(self, v, l=None, init_state=None, n_steps=1, mean_field=False):
        """
        Implements the positive phase sampling, which performs blocks Gibbs
        sampling in order to sample from p(g,h,x,y|v).
        :param v: fixed training set
        :param l: l is None means we sample l, l not None means we clamp l.
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
            init_state['l'] = T.ones((self.batch_size,self.n_l)) * T.nnet.softmax(self.lbias)

        outputs = self.pos_phase(v, l=l,
                init_state=init_state,
                n_steps=n_steps,
                mean_field=mean_field)

        pos_states = OrderedDict()
        pos_states['g'] = outputs[0]
        pos_states['h'] = outputs[1]
        pos_states['l'] = outputs[2] if l is None else self.input_labels

        # update running average of positive phase activations
        pos_updates = OrderedDict()
        pos_updates[self.pos_counter] = outputs[-1]
        pos_updates[self.odd_even] = (self.odd_even + 1) % 2
        return pos_states, pos_updates

    def neg_sampling(self, g_sample, h_sample, v_sample, l_sample, n_steps=1):
        """
        Gibbs step for negative phase, which alternates:
        p(g|b,h,v), p(h|b,g,v), p(b|g,h,v), p(s|b,g,h,v) and p(v|b,g,h,s)
        :param g_sample: T.matrix of shape (batch_size, n_g)
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        :param n_steps: number of Gibbs updates to perform in negative phase.
        """

        def gibbs_iteration(g1, h1, v1, l1):
            g2 = self.sample_g_given_hv(h1, v1)
            h2 = self.sample_h_given_gvl(g2, v1, l1)
            l2 = self.sample_l_given_h(h2)
            s2 = self.sample_s_given_ghv(g2, h2, v1)
            v2 = self.sample_v_given_ghs(g2, h2, s2)
            return [g2, h2, s2, v2, l2]

        [new_g, new_h, new_s, new_v, new_l] , updates = theano.scan(
                gibbs_iteration,
                outputs_info = [g_sample, h_sample, None, v_sample, l_sample],
                n_steps=n_steps)

        final_v = new_v[-1]
        final_g = self.sample_g_given_hv(new_h[-1], final_v)
        final_h = self.sample_h_given_gvl(final_g, final_v, new_l[-1])
        final_l = self.sample_l_given_h(final_h)
        final_s = self.sample_s_given_ghv(final_g, final_h, final_v)

        return [final_g, final_h, final_s, final_v, final_l]

    def neg_sampling_updates(self, n_steps=1, use_pcd=True):
        """
        Implements the negative phase, generating samples from p(h,s,v).
        :param n_steps: scalar, number of Gibbs steps to perform.
        """
        init_chain = self.neg_v if use_pcd else self.input
        [new_g, new_h, new_s, new_v, new_l] =  self.neg_sampling(
                self.neg_g, self.neg_h, self.neg_v, self.neg_l,
                n_steps = n_steps)

        updates = OrderedDict()
        updates[self.neg_g] = new_g
        updates[self.neg_h] = new_h
        updates[self.neg_s] = new_s
        updates[self.neg_v] = new_v
        updates[self.neg_l] = new_l

        return updates

    def ml_cost(self, pos_g, pos_h, pos_v, pos_l, neg_g, neg_h, neg_v, neg_l, mean_field=False):
        """
        Variational approximation to the maximum likelihood positive phase.
        :param v: T.matrix of shape (batch_size, n_v), training examples
        :return: tuple (cost, gradient)
        """
        pos_cost, pos_cte = self.free_energy(pos_g, pos_h, pos_v, pos_l)
        neg_cost, neg_cte = self.free_energy(neg_g, neg_h, neg_v, neg_l)
        cost = pos_cost - neg_cost

        # build gradient of cost with respect to model parameters
        cte = pos_cte + neg_cte

        return costmod.Cost(cost, self.params(), cte)


    ##############################
    # GENERIC OPTIMIZATION STUFF #
    ##############################
    def get_sparsity_cost(self, pos_g, pos_h, pos_l):

        # update mean activation using exponential moving average
        hack_g = self.g_given_hv(pos_h, self.input)
        hack_h = self.h_given_gvl(pos_g, self.input, pos_l)

        # define loss based on value of sp_type
        eps = npy_floatX(1./self.batch_size)
        loss = lambda targ, val: - npy_floatX(targ) * T.log(eps + val) \
                                 - npy_floatX(1-targ) * T.log(1 - val + eps)

        params = []
        cost = T.zeros((), dtype=floatX)
        if self.sp_weight['g'] or self.sp_weight['h']:
            params += [self.Wv, self.Whl, self.alpha, self.mu]
            if self.sp_weight['g']:
                cost += self.sp_weight['g']  * T.sum(loss(self.sp_targ['g'], hack_g.mean(axis=0)))
                params += [self.gbias]
            if self.sp_weight['h']:
                cost += self.sp_weight['h']  * T.sum(loss(self.sp_targ['h'], hack_h.mean(axis=0)))
                params += [self.hbias]
            if self.flags['split_norm']:
                params += [self.scalar_norms]
        cte = [pos_g, pos_h, pos_l]
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

    def dfe_dghat(self, g_hat, h_hat, l_hat, v):
        from_v = self.from_v(v)
        from_h = self.from_h(h_hat)
        rval = self.gbias
        rval += self.to_g(0.5 * 1./self.alpha_prec * from_h * from_v**2)
        rval += self.to_g(self.mu * from_h * from_v)
        dentropy = - (1 - g_hat) * T.xlogx.xlogx(g_hat) + g_hat * T.xlogx.xlogx(1 - g_hat)
        return g_hat * (1-g_hat) * rval  + dentropy

    def dfe_dhhat(self, g_hat, h_hat, l_hat, v):
        from_v = self.from_v(v)
        from_g = self.from_g(g_hat)
        rval = self.hbias
        rval += self.to_h(0.5 * 1./self.alpha_prec * from_g * from_v**2)
        rval += self.to_h(self.mu * from_g * from_v)
        rval += self.label_multiplier * T.dot(l_hat, self.Whl.T)
        dentropy = - (1 - h_hat) * T.xlogx.xlogx(h_hat) + h_hat * T.xlogx.xlogx(1 - h_hat)
        return h_hat * (1-h_hat) * rval  + dentropy

    def dfe_dlhat(self, g_hat, h_hat, l_hat, v):
        # term from loss function
        dloss_dl = self.label_multiplier * (T.dot(h_hat, self.Whl) + self.lbias)
        rval = dloss_dl * l_hat - l_hat * T.shape_padright(T.sum(l_hat * dloss_dl, axis=1))
        # term from entropy.
        # dentropy = T.sum(-l_hat * T.log(l_hat), axis=1)
        dentropy = - T.xlogx.xlogx(l_hat) - l_hat +\
                     l_hat * T.shape_padright(T.sum(T.xlogx.xlogx(l_hat) + l_hat, axis=1))
        return rval + dentropy

    def get_monitoring_channels(self, x, y=None):
        chans = OrderedDict()
        if self.flags['split_norm']:
            chans.update(self.monitor_vector(self.scalar_norms))
        chans.update(self.monitor_matrix(self.Wv))
        chans.update(self.monitor_matrix(self.Wg))
        chans.update(self.monitor_matrix(self.Wh))
        chans.update(self.monitor_matrix(self.Whl))
        chans.update(self.monitor_vector(self.gbias))
        chans.update(self.monitor_vector(self.hbias))
        chans.update(self.monitor_vector(self.lbias))
        chans.update(self.monitor_vector(self.vbias))
        chans.update(self.monitor_vector(self.alpha_prec, name='alpha_prec'))
        chans.update(self.monitor_vector(self.mu))
        chans.update(self.monitor_matrix(self.neg_g))
        chans.update(self.monitor_matrix(self.neg_h))
        chans.update(self.monitor_matrix(self.neg_s - self.mu, name='(neg_s - mu)'))
        chans.update(self.monitor_matrix(self.neg_v))
        chans.update(self.monitor_matrix(self.neg_l))
        wg_norm = T.sqrt(T.sum(self.Wg**2, axis=0))
        wh_norm = T.sqrt(T.sum(self.Wh**2, axis=0))
        whl_norm = T.sqrt(T.sum(self.Whl**2, axis=0))
        wv_norm = T.sqrt(T.sum(self.Wv**2, axis=0))
        chans.update(self.monitor_vector(wg_norm, name='wg_norm'))
        chans.update(self.monitor_vector(wh_norm, name='wh_norm'))
        chans.update(self.monitor_vector(whl_norm, name='whl_norm'))
        chans.update(self.monitor_vector(wv_norm, name='wv_norm'))
        chans['lr'] = self.lr
        # monitor energy vs. label energy
        fe, cte  = self.free_energy(self.neg_g, self.neg_h, self.neg_v, self.neg_l)
        label_fe = T.mean(self.label_energy(self.neg_h, self.neg_l))
        chans['energy'] = fe
        chans['label_energy'] = label_fe

        ### MONITOR MEAN-FIELD CONVERGENCE ###
        pos_states, pos_updates = self.pos_phase_updates(x,
                n_steps = self.pos_steps,
                mean_field=self.flags['mean_field'])
        dfe_dghat = abs(self.dfe_dghat(pos_states['g'], pos_states['h'], pos_states['l'], x))
        dfe_dhhat = abs(self.dfe_dhhat(pos_states['g'], pos_states['h'], pos_states['l'], x))
        dfe_dlhat = abs(self.dfe_dlhat(pos_states['g'], pos_states['h'], pos_states['l'], x))
        chans.update(self.monitor_vector(dfe_dghat, name='abs_dfe_dghat'))
        chans.update(self.monitor_vector(dfe_dhhat, name='abs_dfe_dhhat'))
        chans.update(self.monitor_vector(dfe_dlhat, name='abs_dfe_dlhat'))
        chans['pos_counter'] = self.pos_counter
 
        return chans


class TrainingAlgorithm(default.DefaultTrainingAlgorithm):

    def setup(self, model, dataset):
        x = dataset.get_batch_design(10000, include_labels=False)
        ml_vbias = rbm_utils.compute_ml_bias(x)
        model.vbias.set_value(ml_vbias)
        pval_v = sigm(model.vbias.get_value())
        neg_v = model.rng.binomial(n=1, p=pval_v, size=(model.batch_size, model.n_v))
        model.neg_v.set_value(neg_v.astype(floatX))
        super(TrainingAlgorithm, self).setup(model, dataset)
