import numpy
import theano
import theano.tensor as T

import pylearn2 
import pylearn2.config
from pylearn2.utils import serial
from pylearn2.scripts.jobman.experiment import ydict

dataset =\
    """
    !obj:pylearn2.datasets.mnist_variations.MNIST_variations {{
        "which_set": {which_set},
        "variation": {variation},
        "center": 1,
        "shuffle": 1,
        "one_hot": 1,
    }}
    """

sparse_masks =\
{
'unfactored_g':
    """
    !obj:hossrbm.sparse_masks.sparsity_mask {{
        "type": "unfactored_g",
        "n_g": *n_g,
        "n_h": *n_h,
        "bw_g": &bw_g {bw_g},
        "bw_h": &bw_h {bw_h},
    }}
    """,
'unfactored_h':
    """
    !obj:hossrbm.sparse_masks.sparsity_mask {{
        "type": "unfactored_h",
        "n_g": *n_g,
        "n_h": *n_h,
        "bw_g": *bw_g,
        "bw_h": *bw_h,
    }}
   """
}


svm_callback =\
    """
    !obj:pipelines.svm.svm_on_features.pylearn2_svm_callback {{
        "run_every": 5000,
        "svm": !obj:scikits.learn.svm.LinearSVC {{
            "loss": 'l2',
            "penalty": 'l2',
            # Will be overwritten. Only used for initialization.
            "C": 0.1,
        }},
        "trainset": {trainset},
        "validset": {validset},
        "testset":  {testset},
        "model": *model,
        "model_call_kwargs": {{
            "output_type": {output_type},
        }},
        "C_list": [0.001, 0.01, 0.1, 1.0, 10.],
        "save_fname": 'best_svm.pkl',
    }},
    """

yaml_string = """
!obj:pylearn2.scripts.train.Train {{
    "dataset": &data {dataset},
    "model": &model !obj:hossrbm.hossrbm.BilinearSpikeSlabRBM {{
                "seed" : 123141,
                "init_from" : "",
                "batch_size" : &batch_size {batch_size},
                "n_v"  : {n_v},
                "n_g"  : &n_g {n_g},
                "n_h"  : &n_h {n_h},
                "n_s"  : {n_s},
                "sparse_gmask": {sparse_gmask},
                "sparse_hmask": {sparse_hmask},
                "pos_steps": {pos_steps},
                "neg_sample_steps" : {neg_steps},
                "flags": {{
                    'truncate_s': {truncate_s},
                    'truncate_v': {truncate_v}, 
                    'lambd_interaction': {lambd_interaction},
                    'scalar_lambd': {scalar_lambd},
                    'wg_norm': {wg_norm},
                    'wh_norm': {wh_norm},
                    'wv_norm': {wv_norm},
                    'split_norm': {split_norm},
                    'mean_field': {mean_field},
                }},
                "lr_spec"  : {{
                    'type': 'linear',
                    'start': {lr_start},
                    'end': {lr_end},
                }},
                # WARNING: change default values before
                "lr_timestamp" : [0], # in number of updates
                "lr_mults": {{}},
                "iscales" : {{
                    'Wg': {iscale_wg},
                    'Wh': {iscale_wh},
                    'Wv': {iscale_wv},
                    'scalar_norms': 1.0,
                    'lambd': {iscale_lambd},
                    'mu': {iscale_mu},
                    'alpha': {iscale_alpha},
                    'gbias': {iscale_gbias},
                    'hbias': {iscale_hbias},
                    'vbias': {iscale_vbias},
                }},
                "truncation_bound": {{
                    "v": {truncation_bound_v},
                    "s": {truncation_bound_s},
                }},
                "clip_min" : {{
                    'lambd': {clip_min_lambd},
                    'alpha': {clip_min_alpha},
                }},
                "l1"     : {{}},
                "l2"     : {{}},
                "sp_weight": {{
                    "g": {sp_weight_g},
                    "h": {sp_weight_h},
                }},
                "sp_targ"  : {{
                    "g": {sp_targ_g},
                    "h": {sp_targ_h},
                }},
                "debug": True,
                "save_every": {save_every},
                "save_at": [],
                "max_updates": {max_updates},
                "my_save_path": "model",
    }},
    "algorithm": !obj:ssrbm.pooled_ss_rbm.TrainingAlgorithm {{
               "batch_size": *batch_size,
               "batches_per_iter" : 100,
               "monitoring_batches": 11,
               "monitoring_dataset": *data,
    }},
    "callbacks": [ {callbacks} ],
}}
"""

def experiment(state, channel):
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
    trainset = dataset.format(which_set='train', **state)
    validset = dataset.format(which_set='valid', **state)
    testset  = dataset.format(which_set='test', **state)
    callback = svm_callback.format(trainset=trainset, validset=validset, testset=testset, **state)
    yaml_out = yaml_string.format(
            dataset=trainset,
            callbacks=callback,
            sparse_gmask = sparse_masks[state.sparse_gmask_type].format(**state),
            sparse_hmask = sparse_masks[state.sparse_hmask_type].format(**state),
            **state)

    # generate .yaml file
    fname = 'experiment.yaml'
    fp = open(fname, 'w')
    fp.write(yaml_out)
    fp.close()

    train_obj = pylearn2.config.yaml_parse.load(open(fname,'r'))
    train_obj.model.jobman_channel = channel
    train_obj.model.jobman_state = state
    train_obj.main_loop()
    return channel.COMPLETE
