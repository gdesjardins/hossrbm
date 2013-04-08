"""
This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""
import numpy
import pylab as pl
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
from true_gradient import true_gradient

def sigm(x): return 1./(1 + numpy.exp(-x))
def softplus(x): return numpy.log(1. + numpy.exp(x))
def softplus_inv(x): return numpy.log(numpy.exp(x) - 1.)

class BilinearSpikeSlabRBM(Model, Block):
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
        flags.setdefault('ml_vbias', False)
        flags.setdefault('init_inf_rand', False)
        flags.setdefault('enable_centering', False)
        flags.setdefault('enable_sbias', False)
        flags.setdefault('disable_gbias', False)
        flags.setdefault('disable_hbias', False)
        flags.setdefault('enable_vbias', False)
        flags.setdefault('enable_wvg', False)
        flags.setdefault('enable_wvh', False)
        if len(flags.keys()) != 19:
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
        self.init_centering()

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
        self.Wvg = self.init_weight(self.iscales['Wvg'], (self.n_v, self.n_g), 'Wvg', normalize=False)
        self.Wvh = self.init_weight(self.iscales['Wvh'], (self.n_v, self.n_h), 'Wvh', normalize=False)
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
        norm_wv = numpy.sqrt(numpy.sum(self.Wv.get_value()**2, axis=0)).mean()
        self.avg_norm_wg = sharedX(norm_wg, name='avg_norm_wg')
        self.avg_norm_wh = sharedX(norm_wh, name='avg_norm_wh')
        self.avg_norm_wv = sharedX(norm_wv, name='avg_norm_wv')

        # allocate shared variables for bias parameters
        self.sbias = sharedX(self.iscales['sbias'] * numpy.ones(self.n_s), name='sbias')
        self.gbias = sharedX(self.iscales['gbias'] * numpy.ones(self.n_g), name='gbias')
        self.hbias = sharedX(self.iscales['hbias'] * numpy.ones(self.n_h), name='hbias')
        self.vbias = sharedX(self.iscales['vbias'] * numpy.ones(self.n_v), name='vbias')

        # mean (mu) and precision (alpha) parameters on s
        self.mu = sharedX(self.iscales['mu'] * numpy.ones(self.n_s), name='mu')
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
        # initialize visible unit chains
        scale = numpy.sqrt(1./softplus(self.lambd.get_value()))
        neg_v  = self.rng.normal(loc=0, scale=scale, size=(self.batch_size, self.n_v))
        self.neg_v  = sharedX(neg_v, name='neg_v')
        # initialize s-chain
        loc = self.mu.get_value()
        scale = numpy.sqrt(1./softplus(self.alpha.get_value()))
        neg_s  = self.rng.normal(loc=loc, scale=scale, size=(self.batch_size, self.n_s))
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

    def init_centering(self):
        self.avg_pos_g = sharedX(0.5 * numpy.ones(self.n_g), name='avg_pos_g')
        self.avg_pos_h = sharedX(0.5 * numpy.ones(self.n_h), name='avg_pos_h')
        self.avg_pos_v = sharedX(numpy.zeros(self.n_v), name='avg_pos_v')
        self.avg_pos_g_tm1 = sharedX(0. * numpy.ones(self.n_g), name='avg_pos_g_tm1')
        self.avg_pos_h_tm1 = sharedX(0. * numpy.ones(self.n_h), name='avg_pos_h_tm1')
        self.avg_pos_v_tm1 = sharedX(numpy.zeros(self.n_v), name='avg_pos_v_tm1')
 
    def params(self):
        """
        Returns a list of learnt model parameters.
        """
        params  = [self.Wv]
        if not self.flags['disable_gbias']:
            params += [self.gbias]
        if not self.flags['disable_hbias']:
            params += [self.hbias]
        params += [self.alpha, self.mu, self.lambd]
        if self.flags['split_norm']:
            params += [self.scalar_norms]
        if self.flags['enable_sbias']:
            params += [self.sbias]
        if self.flags['enable_vbias']:
            params += [self.vbias]
        if self.flags['enable_wvg']:
            params += [self.Wvg]
        if self.flags['enable_wvh']:
            params += [self.Wvh]
        return params

    def do_theano(self):
        """ Compiles all theano functions needed to use the model"""

        init_names = dir(self)

        ###### All fields you don't want to get pickled (e.g., theano functions) should be created below this line

        # build energy function
        _gtemp = T.matrix('_gtemp')
        _htemp = T.matrix('_htemp')
        _vtemp = T.matrix('_vtemp')
        energy, cte = self.free_energy(_gtemp, _htemp, _vtemp, mean=False)
        self.energy_func = theano.function([_gtemp, _htemp, _vtemp], energy)

        # SAMPLING: NEGATIVE PHASE
        neg_updates = self.neg_sampling_updates(n_steps=self.neg_sample_steps, use_pcd=True)
        self.sample_func = theano.function([], [], updates=neg_updates)

        # POSITIVE PHASE
        pos_states, pos_updates = self.pos_phase_updates(
                self.input,
                n_steps = self.pos_steps,
                mean_field=self.flags['mean_field'])
        self.inference_func = theano.function([self.input], [pos_states['g'], pos_states['h']])

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
        updates = OrderedDict() 

        if self.flags['scalar_lambd']:
            updates[self.lambd] = T.mean(self.lambd) * T.ones_like(self.lambd)

        if self.Wg in self.params():
            norm_wg = T.sqrt(T.sum(self.Wg**2, axis=0))
            if self.flags['wg_norm'] == 'roland':
                updates[self.Wg] = self.avg_norm_wg * self.Wg / norm_wg
                updates[self.avg_norm_wg] = 0.9 * self.avg_norm_wg + 0.1 * T.mean(norm_wg)
            elif self.flags['wg_norm'] == 'max_unit':
                updates[self.Wg] = self.Wg / norm_wg * T.minimum(norm_wg, 1.0)

        if self.Wh in self.params():
            norm_wh = T.sqrt(T.sum(self.Wh**2, axis=0))
            if self.flags['wh_norm'] == 'roland':
                updates[self.Wh] = self.avg_norm_wh * self.Wh / norm_wh
                updates[self.avg_norm_wh] = 0.9 * self.avg_norm_wh + 0.1 * T.mean(norm_wh)
            elif self.flags['wh_norm'] == 'max_unit':
                updates[self.Wh] = self.Wh / norm_wh * T.minimum(norm_wh, 1.0)

        norm_wv = T.sqrt(T.sum(self.Wv**2, axis=0))
        if self.flags['wv_norm'] == 'max_unit':
            updates[self.Wv] = self.Wv / norm_wv * T.minimum(norm_wv, 1.0)
            updates[self.scalar_norms] = T.maximum(1.0, self.scalar_norms)
        elif self.flags['wv_norm'] == 'max_mean':
            updates[self.Wv] = self.Wv / norm_wv * T.mean(norm_wv)
        elif self.flags['wv_norm'] == 'max_mov_avg':
            updates[self.Wv] = self.Wv / norm_wv * self.avg_norm_wv
            updates[self.avg_norm_wv] = 0.9 * self.avg_norm_wv + 0.1 * T.mean(norm_wv)

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
        if self.batches_seen==0:
            self.debug_init(dataset)
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

    def debug_init(self, dataset):
        x = dataset.get_batch_design(2056, include_labels=False)
        g = numpy.random.randint(low=0, high=2, size=(2056,3)).astype('float32')
        h = numpy.random.randint(low=0, high=2, size=(2056,5)).astype('float32')
        e = self.energy_func(g, h, x)
        pl.hist(e, bins=100)
        pl.show()
        import pdb; pdb.set_trace()

    def free_energy(self, g_sample, h_sample, v_sample, mean=True):
        """
        Computes energy for a given configuration of (g,h,v,x,y).
        :param h_sample: T.matrix of shape (batch_size, n_g)
        :param h_sample: T.matrix of shape (batch_size, n_h)
        :param v_sample: T.matrix of shape (batch_size, n_v)
        """
        cg_sample = self.center_g(g_sample)
        ch_sample = self.center_h(h_sample)
        gh = self.gh_prod(g_sample, h_sample)
        from_v = self.from_v(v_sample)
        energy = -T.sum(0.5 * 1./self.alpha_prec * gh * from_v**2, axis=1)
        energy -= T.sum(self.mu * gh * from_v, axis=1)
        energy += T.sum(0.5 * self.lambd_prec * v_sample**2, axis=1)
        energy -= 0.5 * T.sum(T.log(2*numpy.pi / self.alpha_prec))
        energy -= T.dot(gh, self.sbias)
        energy -= T.dot(cg_sample, self.gbias)
        energy -= T.dot(ch_sample, self.hbias)
        energy -= T.dot(v_sample, self.vbias)
        energy -= T.sum(T.dot(v_sample, self.Wvg) * cg_sample, axis=1)
        energy -= T.sum(T.dot(v_sample, self.Wvh) * ch_sample, axis=1)
        cte  = [g_sample, h_sample, v_sample]
        rval = T.mean(energy) if mean else energy
        return rval, cte

    def __call__(self, v, output_type='g+h', mean_field=True):
        print 'Building representation with %s' % output_type
        init_state = OrderedDict()
        init_state['g'] = T.ones((v.shape[0],self.n_g)) * T.nnet.sigmoid(self.gbias)
        init_state['h'] = T.ones((v.shape[0],self.n_h)) * T.nnet.sigmoid(self.hbias)
        [g, h, pos_counter] = self.pos_phase(v, init_state, n_steps=self.pos_steps, mean_field=mean_field)
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

    def center_g(self, g_sample):
        if self.flags['enable_centering']:
            g_sample -= self.avg_pos_g
        return g_sample
    
    def center_h(self, h_sample):
        if self.flags['enable_centering']:
            h_sample -= self.avg_pos_h
        return h_sample

    def gh_prod(self, g_sample, h_sample):
        return self.from_g(g_sample) * self.from_h(h_sample)

    def from_v(self, v_sample):
        Wv = self.Wv
        if self.flags['split_norm']:
            Wv *= self.scalar_norms
        if self.flags['lambd_interaction']:
            temp = self.lambd_prec * v_sample
        else:
            temp = v_sample
        return T.dot(temp - self.vbias, Wv)

    def from_g(self, g_sample):
        return T.dot(self.center_g(g_sample), self.Wg)

    def from_h(self, h_sample):
        return T.dot(self.center_h(h_sample), self.Wh)

    def to_g(self, g_s):
        return T.dot(g_s, self.Wg.T)

    def to_h(self, h_s):
        return T.dot(h_s, self.Wh.T)

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
        h_mean_s += from_g * self.sbias
        h_mean  = self.to_h(h_mean_s)
        h_mean += T.dot(v_sample, self.Wvh) + self.hbias
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
        g_mean_s += from_h * self.sbias
        g_mean = self.to_g(g_mean_s)
        g_mean += T.dot(v_sample, self.Wvg) + self.gbias
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
        gh = self.gh_prod(g_sample, h_sample)
        from_v = self.from_v(v_sample)
        s_mean = (1./self.alpha_prec * from_v + self.mu) * gh
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
        cg_sample = self.center_g(g_sample)
        ch_sample = self.center_h(h_sample)
        gh = self.gh_prod(g_sample, h_sample)
        Wv = self.Wv
        if self.flags['split_norm']:
            Wv *= self.scalar_norms
        v_mean = T.dot(s_sample * gh, Wv.T) + self.vbias
        v_mean += T.dot(cg_sample, self.Wvg.T)
        v_mean += T.dot(ch_sample, self.Wvh.T)
        if not self.flags['lambd_interaction']:
            v_mean *= 1./self.lambd_prec
        return v_mean

    def sample_v_given_ghs(self, g_sample, h_sample, s_sample, rng=None, size=None):
        v_mean = self.v_given_ghs(g_sample, h_sample, s_sample)

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
    def pos_phase(self, v, init_state, n_steps=1, mean_field=False, eps=1e-5):
        """
        Mixed mean-field + sampling inference in positive phase.
        :param v: input being conditioned on
        :param init: dictionary of initial values
        :param n_steps: number of Gibbs updates to perform afterwards.
        """
        def pos_gibbs_iteration(g1, h1, pos_counter, v, size):
            g2 = self.sample_g_given_hv(h1, v, size=size)
            h2 = self.sample_h_given_gv(g2, v, size=size)
            return [g2, h2, pos_counter + 1]

        def pos_mf_iteration(g1, h1, pos_counter, v, size):
            # branch1: start with g
            branch1_g2 = self.g_given_hv(h1, v)
            branch1_h2 = self.h_given_gv(branch1_g2, v)
            # branch2: start with g
            branch2_h2 = self.h_given_gv(g1, v)
            branch2_g2 = self.g_given_hv(branch2_h2, v)
            # decide which way we should sample
            g2 = ifelse(self.odd_even, branch1_g2, branch2_g2)
            h2 = ifelse(self.odd_even, branch1_h2, branch2_h2)
            # stopping criterion
            dl_dghat = T.sum(self.dfe_dghat(g2, h2, v)**2)
            dl_dhhat = T.sum(self.dfe_dhhat(g2, h2, v)**2)
            stop = T.sqrt(dl_dghat + dl_dhhat)
            return [g2, h2, pos_counter + 1], theano.scan_module.until(stop < eps)
            #return [g2, h2, pos_counter + 1]

        [new_g, new_h, pos_counter], updates = theano.scan(
                pos_mf_iteration if mean_field else pos_gibbs_iteration,
                outputs_info = [init_state['g'], init_state['h'], 0.],
                non_sequences = [v, v.shape[0]],
                n_steps=n_steps)

        new_g = new_g[-1]
        new_h = new_h[-1]
        new_pos_counter = pos_counter[-1]

        return [new_g, new_h, new_pos_counter]

    def pos_phase_updates(self, v, init_state=None, n_steps=1, mean_field=False, alpha=0.):
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
            _g = 0.5 * T.ones((self.batch_size,self.n_g))
            _h = 0.5 * T.ones((self.batch_size,self.n_h))
            init_state['g'] = self.g_given_hv(_h, v)
            init_state['h'] = self.h_given_gv(_g, v)

        [new_g, new_h, pos_counter] = self.pos_phase(v,
                init_state=init_state,
                n_steps=n_steps,
                mean_field=mean_field)

        pos_states = OrderedDict()
        pos_states['g'] = new_g
        pos_states['h'] = new_h

        # update running average of positive phase activations
        pos_updates = OrderedDict()
        pos_updates[self.pos_counter] = pos_counter
        pos_updates[self.odd_even] = (self.odd_even + 1) % 2
        pos_updates[self.pos_g] = new_g
        pos_updates[self.pos_h] = new_h
        if self.flags['enable_centering']:
            pos_updates[self.avg_pos_g] = (1.-alpha) * self.avg_pos_g + alpha * T.mean(new_g, axis=0)
            pos_updates[self.avg_pos_h] = (1.-alpha) * self.avg_pos_h + alpha * T.mean(new_h, axis=0)
            pos_updates[self.avg_pos_v] = (1.-alpha) * self.avg_pos_v + alpha * T.mean(v, axis=0)

        pos_updates[self.avg_pos_g_tm1] = self.avg_pos_g
        pos_updates[self.avg_pos_h_tm1] = self.avg_pos_h
        pos_updates[self.avg_pos_v_tm1] = self.avg_pos_v

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

    def dfe_dghat(self, g_hat, h_hat, v):
        from_v = self.from_v(v)
        from_h = self.from_h(h_hat)
        rval_s = 0.5 * 1./self.alpha_prec * from_h * from_v**2
        rval_s += self.mu * from_h * from_v
        rval_s += from_h * self.sbias
        rval = self.to_g(rval_s)
        rval += T.dot(v, self.Wvg)
        rval += self.gbias
        dentropy = - (1 - g_hat) * T.xlogx.xlogx(g_hat) + g_hat * T.xlogx.xlogx(1 - g_hat)
        return g_hat * (1-g_hat) * rval  + dentropy

    def dfe_dhhat(self, g_hat, h_hat, v):
        from_v = self.from_v(v)
        from_g = self.from_g(g_hat)
        rval_s = 0.5 * 1./self.alpha_prec * from_g * from_v**2
        rval_s += self.mu * from_g * from_v
        rval_s += from_g * self.sbias
        rval = self.to_h(rval_s)
        rval += T.dot(v, self.Wvh)
        rval += self.hbias
        dentropy = - (1 - h_hat) * T.xlogx.xlogx(h_hat) + h_hat * T.xlogx.xlogx(1 - h_hat)
        return h_hat * (1-h_hat) * rval  + dentropy

    def get_monitoring_channels(self, x, y=None):
        chans = OrderedDict()
        if self.flags['split_norm']:
            chans.update(self.monitor_vector(self.scalar_norms))

        gh = self.gh_prod(self.pos_g, self.pos_h)
        chans.update(self.monitor_matrix(self.Wv))
        chans.update(self.monitor_matrix(self.Wvg))
        chans.update(self.monitor_matrix(self.Wvh))
        chans.update(self.monitor_matrix(self.Wg))
        chans.update(self.monitor_matrix(self.Wh))
        chans.update(self.monitor_vector(self.gbias))
        chans.update(self.monitor_vector(self.hbias))
        chans.update(self.monitor_vector(self.sbias))
        chans.update(self.monitor_vector(self.vbias))
        chans.update(self.monitor_vector(self.alpha_prec, name='alpha_prec'))
        chans.update(self.monitor_vector(self.mu))
        chans.update(self.monitor_vector(self.lambd_prec, name='lambd_prec'))
        chans.update(self.monitor_matrix(self.pos_g))
        chans.update(self.monitor_matrix(self.pos_h))
        chans.update(self.monitor_matrix(self.neg_g))
        chans.update(self.monitor_matrix(self.neg_h))
        chans.update(self.monitor_matrix(self.neg_s - self.mu, name='(neg_s - mu)'))
        chans.update(self.monitor_matrix(self.neg_v))
        chans.update(self.monitor_vector(self.avg_pos_g))
        chans.update(self.monitor_vector(self.avg_pos_h))
        chans.update(self.monitor_vector(self.avg_pos_v))
        wv_norm = T.sqrt(T.sum(self.Wv**2, axis=0))
        chans.update(self.monitor_vector(wv_norm, name='wv_norm'))
        chans['lr'] = self.lr

        ### MONITOR MEAN-FIELD CONVERGENCE ###
        pos_states, pos_updates = self.pos_phase_updates(x,
                n_steps = self.pos_steps,
                mean_field=self.flags['mean_field'])
        dfe_dghat = abs(self.dfe_dghat(pos_states['g'], pos_states['h'], x))
        dfe_dhhat = abs(self.dfe_dhhat(pos_states['g'], pos_states['h'], x))
        chans.update(self.monitor_vector(dfe_dghat, name='abs_dfe_dghat'))
        chans.update(self.monitor_vector(dfe_dhhat, name='abs_dfe_dhhat'))
        chans['pos_counter'] = self.pos_counter
        return chans


class TrainingAlgorithm(default.DefaultTrainingAlgorithm):

    def setup(self, model, dataset):
        x = dataset.get_batch_design(10000, include_labels=False)
        if model.flags['ml_lambd']:
            scale = (1./numpy.std(x, axis=0))**2
            model.lambd.set_value(softplus_inv(scale).astype(floatX))
            # reset neg_v markov chain accordingly
            neg_v = model.rng.normal(loc=0, scale=scale, size=(model.batch_size, model.n_v))
            model.neg_v.set_value(neg_v.astype(floatX))
        if model.flags['ml_vbias']:
            model.vbias.set_value(x.mean(axis=0))
        super(TrainingAlgorithm, self).setup(model, dataset)
