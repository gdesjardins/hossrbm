import numpy
import theano
import theano.tensor as T

import pylearn2 
import pylearn2.config
from pylearn2.utils import serial
from pylearn2.scripts.jobman.experiment import ydict

dataset =\
    """
    &data !obj:pylearn2.datasets.mnist_variations.MNIST_variations {
            "which_set": %(which_set)s,
            "variation": &variation %(variation)s,
            "center": 1,
            "shuffle": 1,
            "one_hot": 1,
        }
    """

sparse_masks =\
{
'unfactored_g':
    """
    !obj:hossrbm.sparse_masks.sparsity_mask {
        "type": "unfactored_g",
        "n_g": *n_g,
        "n_h": *n_h,
        "bw_g": &bw_g %(bw_g)i,
        "bw_h": &bw_h %(bw_h)i,
    }
    """,
'unfactored_h':
    """
    !obj:hossrbm.sparse_masks.sparsity_mask {
        "type": "unfactored_h",
        "n_g": *n_g,
        "n_h": *n_h,
        "bw_g": *bw_g,
        "bw_h": *bw_h,
    }
   """
}


svm_callback =\
    """
    !obj:pipelines.svm.svm_on_features.pylearn2_svm_callback {
        "run_every": 5000,
        "svm": !obj:scikits.learn.svm.LinearSVC {
            "loss": 'l2',
            "penalty": 'l2',
            # Will be overwritten. Only used for initialization.
            "C": 0.1,
        },
        "trainset": %(trainset)s,
        "validset": %(validset)s,
        "testset":  %(testset)s,
        "model": *model,
        "model_call_kwargs": {
            "output_type": %(output_type)s,
        },
        "C_list": [0.001, 0.01, 0.1, 1.0, 10.],
        "save_fname": 'best_svm.pkl',
    },
    """

yaml_string = """
!obj:pylearn2.scripts.train.Train {
    "dataset": %(dataset)s,
    "model": &model !obj:hossrbm.hossrbm.BilinearSpikeSlabRBM {
                "seed" : 123141,
                "init_from" : "",
                "batch_size" : &batch_size %(batch_size)i,
                "n_v"  : %(n_v)i,
                "n_g"  : &n_g %(n_g)i,
                "n_h"  : &n_h %(n_h)i,
                "n_s"  : %(n_s)i,
                "sparse_gmask": %(sparse_gmask)s,
                "sparse_hmask": %(sparse_hmask)s,
                "pos_steps": %(pos_steps),
                "neg_sample_steps" : %(neg_steps)i,
                "flags": {
                    'truncate_s': %(truncate_s)i,
                    'truncate_v': %(truncate_v)i, 
                    'lambd_interaction': %(lambd_interaction)i,
                    'scalar_lambd': %(scalar_lambd)i,
                    'wg_norm': %(wg_norm)s,
                    'wh_norm': %(wh_norm)s,
                    'wv_norm': %(wv_norm)s,
                    'split_norm': %(split_norm)i,
                    'mean_field': %(mean_field)i,
                },
                "lr_spec"  : {
                    'type': 'linear',
                    'start': %(lr_start)f,
                    'end': %(lr_end)f,
                },
                # WARNING: change default values before
                "lr_timestamp" : [0], # in number of updates
                "lr_mults": {},
                "iscales" : {
                    'Wg': %(iscale_wg)f,
                    'Wh': %(iscale_wh)f,
                    'Wv': %(iscale_wv)f,
                    'scalar_norms': 1.0,
                    'lambd': %(iscale_lambd)f,
                    'mu': %(iscale_mu)f,
                    'alpha': %(iscale_alpha)f,
                    'gbias': %(iscale_gbias)f,
                    'hbias': %(iscale_hbias)f,
                    'vbias': %(iscale_vbias)f,
                },
                "truncation_bound": {
                    "v": %(truncation_bound_v)f,
                    "s": %(truncation_bound_s)f,
                },
                "clip_min" : {
                    'lambd': %(clip_min_lambd)f,
                    'alpha': %(clip_min_alpha)f,
                },
                "l1"     : {},
                "l2"     : {},
                "sp_weight": {
                    "g": %(sp_weight_g)f,
                    "h": %(sp_weight_h)f,
                },
                "sp_targ"  : {
                    "g": %(sp_targ_g)f,
                    "h": %(sp_targ_h)f,
                },
                "debug": True,
                "save_every": %(save_every)i,
                "save_at": [],
                "max_updates": %(max_updates)i,
                "my_save_path": "model",
    },
    "algorithm": !obj:ssrbm.pooled_ss_rbm.TrainingAlgorithm {
               "batch_size": *batch_size,
               "batches_per_iter" : 100,
               "monitoring_batches": 11,
               "monitoring_dataset": *data,
    },
    "callbacks": [ %(callbacks)s ],
}
"""

def train_experiment(state, channel):
    """
    Train a model specified in state, and extract required results.

    This function builds a YAML string from ``state.yaml_template``, taking
    the values of hyper-parameters from ``state.hyper_parameters``, creates
    the corresponding object and trains it (like train.py), then run the
    function in ``state.extract_results`` on it, and store the returned values
    into ``state.results``.

    To know how to use this function, you can check the example in tester.py
    (in the same directory).
    """

    # update base yaml config with jobman commands
    trainset = dataset % {'which_set': 'train'}.update(state)
    validset = dataset % {'which_set': 'valid'}.update(state)
    testset  = dataset % {'which_set': 'test'}.update(state)
    callback = callback % {'trainset': trainset, 'validset': validset, 'testset': testset}
    yaml_string = yaml_string % state
    yaml_string = yaml_string % {'callbacks': callback}
    yaml_string = yaml_string % {
        'sparse_gmask': sparse_masks[state.sparse_gmask_type] % state,
        'sparse_hmask': sparse_masks[state.sparse_hmask_type] % state,
    }

    # generate .yaml file
    fname = 'experiment.yaml'
    fp = open(fname, 'w')
    fp.write(yaml_string)
    fp.close()

    train_obj = pylearn2.config.yaml_parse.load(open(fname,'r'))
    train_obj.model.jobman_channel = channel
    train_obj.model.jobman_state = state
    train_obj.main_loop()
    return channel.COMPLETE
