"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

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

from utils import tools
from hossrbm import costmod
from utils import sharedX, floatX, npy_floatX
import truncated

def sigm(x): return 1./(1 + numpy.exp(-x))
def softplus(x): return numpy.log(1. + numpy.exp(x))
def softplus_inv(x): return numpy.log(numpy.exp(x) - 1.)

class PooledSpikeSlabRBM(Model, Block):
    """Spike & Slab Restricted Boltzmann Machine (RBM)  """

    def load_params(self, model):
        fp = open(model)
        model = pickle.load(fp)
        fp.close()

        self.Wv.set_value(model.Wv.get_value())
        self.hbias.set_value(model.hbias.get_value())
        self.mu.set_value(model.mu.get_value())
        self.alpha.set_value(model.alpha.get_value())
        self.lambd.set_value(model.lambd.get_value())
        self.scalar_norms.set_value(model.scalar_norms.get_value())
        # sync negative phase particles
        self.neg_v.set_value(model.neg_v.get_value())
        self.neg_s.set_value(model.neg_s.get_value())
        self.neg_h.set_value(model.neg_h.get_value())
        self.sp_pos_v.set_value(model.sp_pos_v.get_value())
        self.sp_pos_h.set_value(model.sp_pos_h.get_value())
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
        flags.setdefault('split_norm', False)
        flags.setdefault('use_cd', False)
        flags.setdefault('use_energy', False)
        flags.setdefault('wv_norm', None)
        flags.setdefault('truncated_normal', False)
        flags.setdefault('lambd_interaction', False)
        flags.setdefault('scalar_lambd', False)
        flags.setdefault('ml_lambd', False)
        if len(flags.keys()) != 8:
            raise NotImplementedError('One or more flags are currently not implemented.')

    def __init__(self, numpy_rng = None, theano_rng = None,
            n_h=100, bw_s=1, n_v=100, init_from=None,
            neg_sample_steps=1,
            lr_spec=None, lr_timestamp=None, lr_mults = {},
            iscales={}, clip_min={}, clip_max={}, truncation_bound={},
            l1 = {}, l2 = {}, orth_lambda=0.,
            var_param_alpha='exp', var_param_lambd='linear',
            sp_type='kl', sp_weight={}, sp_targ={},
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
        for k in ['Wv', 'hbias']: assert k in iscales.keys()
        iscales.setdefault('mu', 1.)
        iscales.setdefault('alpha', 0.)
        iscales.setdefault('lambd', 0.)
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
   
    def init_parameters(self):
        self.n_s = self.n_h * self.bw_s
        self.scalar_norms = sharedX(1.0 * numpy.ones(self.n_s), name='scalar_norms')
        wv_val =  self.rng.randn(self.n_v, self.n_s) * self.iscales['Wv']
        self.Wv = sharedX(wv_val, name='Wv')
        self.Wh = numpy.zeros((self.n_h, self.n_s), dtype=floatX)
        for i in xrange(self.n_h):
            self.Wh[i, i*self.bw_s:(i+1)*self.bw_s] = 1.

        # allocate shared variables for bias parameters
        self.hbias = sharedX(self.iscales['hbias'] * numpy.ones(self.n_h), name='hbias') 

        # mean (mu) and precision (alpha) parameters on s
        self.mu = sharedX(self.iscales['mu'] * numpy.ones(self.n_s), name='mu')
        self.alpha = sharedX(self.iscales['alpha'] * numpy.ones(self.n_s), name='alpha')
        self.alpha_prec = T.nnet.softplus(self.alpha)

        # diagonal of precision matrix of visible units
        self.lambd = sharedX(self.iscales['lambd'] * numpy.ones(self.n_v), name='lambd')
        self.lambd_prec = T.nnet.softplus(self.lambd)

    def init_chains(self):
        """ Allocate shared variable for persistent chain """
        # initialize visible unit chains
        scale = numpy.sqrt(1./softplus(self.lambd.get_value()))
        neg_v  = self.rng.normal(loc=0, scale=scale, size=(self.batch_size, self.n_v))
        self.neg_v  = sharedX(neg_v, name='neg_v')
        # initialize s-chain
        loc = self.mu.get_value()
        scale = numpy.sqrt(1./softplus(self.alpha.get_value()))
        neg_s  = self.rng.normal(loc=loc, scale=scale, size=(self.batch_size, self.n_s))
        self.neg_s  = sharedX(neg_s, name='neg_s')
        # initialize binary h chains
        pval_h = sigm(self.hbias.get_value())
        neg_h = self.rng.binomial(n=1, p=pval_h, size=(self.batch_size, self.n_h))
        self.neg_h  = sharedX(neg_h, name='neg_h')
        # moving average values for sparsity
        self.sp_pos_v = sharedX(neg_v, name='sp_pos_v')
        self.sp_pos_h = sharedX(neg_h, name='sp_pos_h')
 
    def params(self):
        """
        Returns a list of learnt model parameters.
        """
        params = [self.Wv, self.hbias, self.alpha, self.mu, self.lambd]
        if self.flags['split_norm']:
            params += [self.scalar_norms]
        return params

    def do_theano(self):
        """ Compiles all theano functions needed to use the model"""

        init_names = dir(self)

        ###### All fields you don't want to get pickled (e.g., theano functions) should be created below this line
        # SAMPLING: NEGATIVE PHASE
        neg_updates = self.neg_sampling_updates(
                n_steps=self.neg_sample_steps,
                use_pcd=not self.flags['use_cd'])
        self.sample_func = theano.function([], [], updates=neg_updates)

        # determing maximum likelihood cost
        ml_cost = self.ml_cost(pos_v = self.input, neg_v = neg_updates[self.neg_v])
        main_cost = [ml_cost,
                     self.get_sparsity_cost(),
                     self.get_reg_cost(self.l2, self.l1)]

        ##
        # COMPUTE GRADIENTS WRT. TO ALL COSTS
        ##
        learning_grads = costmod.compute_gradients(self.lr, self.lr_mults, *main_cost)

        ##
        # BUILD UPDATES DICTIONARY
        ##
        learning_updates = costmod.get_updates(learning_grads)
        learning_updates.update(neg_updates)
        learning_updates.update({self.iter: self.iter+1})
      
        # build theano function to train on a single minibatch
        self.batch_train_func = function([self.input], [],
                                         updates=learning_updates,
                                         name='train_rbm_func')


        # enforce constraints function
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

        # constraint filters to have unit norm
        if self.flags['wv_norm'] in ('unit', 'max_unit'):
            wv = constraint_updates.get(self.Wv, self.Wv)
            wv_norm = T.sqrt(T.sum(wv**2, axis=0))
            if self.flags['wv_norm'] == 'unit':
                constraint_updates[self.Wv] = wv / wv_norm
            elif self.flags['wv_norm'] == 'max_unit':
                constraint_updates[self.Wv] = wv / wv_norm * T.minimum(wv_norm, 1.0)

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
        if self.flags['truncated_normal']:
            x = numpy.clip(x, -self.truncation_bound['v'], self.truncation_bound['v'])
        self.batch_train_func(x)

        # accounting...
        self.examples_seen += self.batch_size
        self.batches_seen += 1

        self.enforce_constraints()

        # save to different path each epoch
        if self.my_save_path and \
           (self.batches_seen in self.save_at or
            self.batches_seen % self.save_every == 0):
            fname = self.my_save_path + '_e%i.pkl' % self.batches_seen
            print 'Saving to %s ...' % fname,
            serial.save(fname, self)
            print 'done'

        return self.batches_seen < self.max_updates

    def from_v(self, v_sample):
        Wv = self.Wv
        if self.flags['split_norm']:
            Wv *= self.scalar_norms
        if self.flags['lambd_interaction']:
            temp = self.lambd_prec * v_sample
        else:
            temp = v_sample
        return T.dot(temp, Wv)

    def from_h(self, h_sample):
        return T.dot(h_sample, self.Wh)

    def energy(self, h_sample, s_sample, v_sample):
        """
        Computes energy for a given configuration of (g,h,v,x,y).
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param s_sample: T.matrix of shape (batch_size, n_s)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        """
        from_v = self.from_v(v_sample)
        from_h = self.from_h(h_sample)
        energy = -T.sum(s_sample * from_v * from_h, axis=1)
        energy += T.sum(0.5 * self.alpha_prec * s_sample**2, axis=1)
        energy += T.sum(0.5 * self.lambd_prec * v_sample**2, axis=1)
        energy -= T.sum(self.alpha_prec * self.mu * s_sample * from_h, axis=1)
        energy += T.sum(0.5 * self.alpha_prec * self.mu**2 * from_h, axis=1)
        energy -= T.dot(h_sample, self.hbias)
        return energy

    def free_energy(self, v_sample):
        fe = T.sum(0.5 * self.lambd_prec * v_sample**2, axis=1)
        fe -= 0.5 * T.sum(T.log(2*numpy.pi / self.alpha_prec))
        h_mean = self.h_given_v_input(v_sample)
        fe -= T.sum(T.nnet.softplus(h_mean), axis=1)
        return fe

    def __call__(self, v, output_type='h'):
        assert output_type in ['h', 'hs']
        h_mean = self.h_given_v(v)
        s_mean = self.s_given_hv(h_mean, v)
        output_prods = {
                'h': h_mean,
                'hs': h_mean * T.sqrt(T.dot(s_mean**2, self.Wh.T))
                }
        return output_prods[output_type]

    ######################################
    # MATH FOR CONDITIONAL DISTRIBUTIONS #
    ######################################

    def h_given_v_input(self, v_sample):
        """
        Compute mean activation of h given v.
        :param v_sample: T.matrix of shape (batch_size, n_v matrix)
        """
        Wv = self.scalar_norms * self.Wv
        from_v = T.dot(v_sample, Wv)
        s_mean = 0.5 * 1./self.alpha_prec * from_v**2
        s_mean += from_v * self.mu
        h_mean = T.dot(s_mean, self.Wh.T) + self.hbias
        return h_mean
    
    def h_given_v(self, v_sample):
        h_mean = self.h_given_v_input(v_sample)
        return T.nnet.sigmoid(h_mean)

    def sample_h_given_v(self, v_sample, rng=None):
        """
        Generates sample from p(h|v)
        """
        h_mean = self.h_given_v(v_sample)

        rng = self.theano_rng if rng is None else rng
        h_sample = rng.binomial(size=(self.batch_size,self.n_h),
                                            n=1, p=h_mean, dtype=floatX)
        return h_sample

    def s_given_hv(self, h_sample, v_sample):
        Wv = self.scalar_norms * self.Wv
        from_v = self.from_v(v_sample)
        from_h = self.from_h(h_sample)
        s_mean = (1./self.alpha_prec * from_v + self.mu) * from_h
        return s_mean

    def sample_s_given_hv(self, h_sample, v_sample, rng=None):
        s_mean = self.s_given_hv(h_sample, v_sample)
        
        rng = self.theano_rng if rng is None else rng
        s_sample = rng.normal(
                size=(self.batch_size, self.n_s),
                avg = s_mean, 
                std = T.sqrt(1./self.alpha_prec), dtype=floatX)
        return s_sample

    def v_given_hs(self, h_sample, s_sample):
        """
        Computes the mean-activation of visible units, given all other variables.
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param s_sample: T.matrix of shape (batch_size, n_s)
        """
        Wv = self.scalar_norms * self.Wv
        from_h = self.from_h(h_sample)
        v_mean = T.dot(s_sample * from_h, Wv.T)
        if not self.flags['lambd_interaction']:
            v_mean *= 1./self.lambd_prec
        return v_mean

    def sample_v_given_hs(self, h_sample, s_sample, rng=None):
        v_mean = self.v_given_hs(h_sample, s_sample)

        rng = self.theano_rng if rng is None else rng
        if self.flags['truncated_normal']:
            v_sample = truncated.truncated_normal(
                    size=(self.batch_size, self.n_v),
                    avg = v_mean, 
                    std = T.sqrt(1./self.lambd_prec),
                    lbound = -self.truncation_bound['v'],
                    ubound =  self.truncation_bound['v'],
                    theano_rng = rng,
                    dtype=floatX)
        else:
            v_sample = rng.normal(
                    size=(self.batch_size, self.n_v),
                    avg = v_mean, 
                    std = T.sqrt(1./self.lambd_prec), dtype=floatX)
        return v_sample

    def do_debug(self):
        rng = RandomStreams(seed=12312)
        in1 = T.matrix()
        in2 = T.matrix()
        self.h_given_v_func = theano.function([in1], self.h_given_v(in1))
        self.sample_h_given_v_func = theano.function([in1], self.sample_h_given_v(in1, rng=rng))
        self.s_given_hv_func  = theano.function([in1, in2], self.s_given_hv(in1, in2))
        self.sample_s_given_hv_func = theano.function([in1, in2], self.sample_s_given_hv(in1, in2, rng=rng))
        self.v_given_hs_func = theano.function([in1, in2], self.v_given_hs(in1, in2))
        self.sample_v_given_hs_func = theano.function([in1, in2], self.sample_v_given_hs(in1, in2, rng=rng))

    ##################
    # SAMPLING STUFF #
    ##################

    def neg_sampling(self, v_sample, n_steps=1):
        """
        Gibbs step for negative phase, which alternates: 
        p(h|v), p(s|h,v) and p(v|h,s)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        :param n_steps: number of Gibbs updates to perform in negative phase.
        """

        def gibbs_iteration(v1):
            h2 = self.sample_h_given_v(v1)
            s2 = self.sample_s_given_hv(h2, v1)
            v2 = self.sample_v_given_hs(h2, s2)
            return [h2, s2, v2]

        [new_h, new_s, new_v] , updates = theano.scan(
                gibbs_iteration,
                outputs_info = [None, None, v_sample],
                n_steps=n_steps)

        return [new_h[-1], new_s[-1], new_v[-1]]

    def neg_sampling_updates(self, n_steps=1, use_pcd=True):
        """
        Implements the negative phase, generating samples from p(h,s,v).
        :param n_steps: scalar, number of Gibbs steps to perform.
        """
        init_chain = self.neg_v if use_pcd else self.input
        [new_h, new_s, new_v] =  self.neg_sampling(init_chain, n_steps=n_steps)

        # we want to plot the expected value of the samples
        new_ev = self.v_given_hs(new_h, new_s)

        updates = OrderedDict()
        updates[self.neg_h] = new_h
        updates[self.neg_s] = new_s
        updates[self.neg_v] = new_v
        return updates

    def ml_cost(self, pos_v, neg_v):
        pos_cost = T.sum(self.free_energy(pos_v))
        neg_cost = T.sum(self.free_energy(neg_v))
        batch_cost = pos_cost - neg_cost
        cost = batch_cost / self.batch_size
        return costmod.Cost(cost, self.params(), [pos_v,neg_v])

    def get_sparsity_cost(self):

        # update mean activation using exponential moving average
        hack_h = self.h_given_v(self.sp_pos_v)

        # define loss based on value of sp_type
        if self.sp_type == 'kl':
            eps = npy_floatX(1./self.batch_size)
            loss = lambda targ, val: - npy_floatX(targ) * T.log(eps + val) \
                                     - npy_floatX(1-targ) * T.log(1 - val + eps)
        else:
            raise NotImplementedError('Sparsity type %s is not implemented' % self.sp_type)

        cost = T.zeros((), dtype=floatX)

        params = []
        if self.sp_weight['h']: 
            cost += self.sp_weight['h']  * T.sum(loss(self.sp_targ['h'], hack_h.mean(axis=0)))
            params += [self.hbias]

        if self.sp_type in ['kl'] and self.sp_weight['h']:
            params += [self.Wv, self.alpha, self.mu]
            if self.flags['split_norm']:
                params += [self.scalar_norms]

        return costmod.Cost(cost, params)

    ##############################
    # GENERIC OPTIMIZATION STUFF #
    ##############################
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
        chans.update(self.monitor_matrix(self.Wv))
        chans.update(self.monitor_vector(self.scalar_norms))
        chans.update(self.monitor_vector(self.hbias))
        chans.update(self.monitor_vector(self.alpha_prec, name='alpha_prec'))
        chans.update(self.monitor_vector(self.mu))
        chans.update(self.monitor_vector(self.lambd_prec, name='lambd_prec'))
        chans.update(self.monitor_matrix(self.neg_h))
        chans.update(self.monitor_matrix(self.neg_s))
        chans.update(self.monitor_matrix(self.neg_v))
        scalar_norms = T.sqrt(T.sum(self.Wv**2, axis=0))
        chans['wv_norm.mean'] = T.mean(scalar_norms)
        chans['wv_norm.max'] = T.max(scalar_norms)
        chans['wv_norm.min'] = T.min(scalar_norms)
        chans['lr'] = self.lr
        return chans


class TrainingAlgorithm(default.DefaultTrainingAlgorithm):

    def setup(self, model, dataset):
        if model.flags['ml_lambd']:
            # compute maximum likelihood solution for lambd
            x = dataset.get_batch_design(10000, include_labels=False)
            scale = (1./numpy.std(x, axis=0))**2
            model.lambd.set_value(softplus_inv(scale).astype(floatX))
            # reset neg_v markov chain accordingly
            neg_v = model.rng.normal(loc=0, scale=scale, size=(model.batch_size, model.n_v))
            model.neg_v.set_value(neg_v.astype(floatX))
        super(TrainingAlgorithm, self).setup(model, dataset)
