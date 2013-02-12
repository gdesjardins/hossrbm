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
        "center": {center},
        "shuffle": 1,
        "one_hot": 1,
        "gcn": {gcn},
        "standardize_pixels": {standardize_pixels},
    }}
    """

svm_callback =\
    """
    !obj:pipelines.svm.svm_on_features.pylearn2_svm_callback {{
        "retrain_on_valid": 0,
        "run_every": 50000,
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
        "C_list": [0.0001,0.001, 0.01, 0.1, 1.0, 10.],
        "save_fname": 'best_svm.pkl',
    }},
    """

yaml_string = """
!obj:pylearn2.scripts.train.Train {{
    "dataset": &data {dataset},
    "model": &model !obj:hossrbm.pooled_ss_rbm.PooledSpikeSlabRBM {{
                "seed" : 123141,
                "init_from" : "",
                "batch_size" : &batch_size {batch_size},
                "n_v"  : {n_v},
                "n_h"  : &n_h {n_h},
                "bw_s" : {bw_s},
                "neg_sample_steps" : {neg_steps},
                "flags": {{
                    'truncate_v': {truncate_v}, 
                    'lambd_interaction': {lambd_interaction},
                    'scalar_lambd': {scalar_lambd},
                    'wv_norm': {wv_norm},
                    'split_norm': {split_norm},
                    'ml_lambd': {ml_lambd},
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
                    'Wv': {iscale_wv},
                    'scalar_norms': 1.0,
                    'lambd': {iscale_lambd},
                    'mu': {iscale_mu},
                    'alpha': {iscale_alpha},
                    'hbias': {iscale_hbias},
                }},
                "truncation_bound": {{
                    "v": {truncation_bound_v},
                }},
                "clip_min" : {{
                    'lambd': {clip_min_lambd},
                    'alpha': {clip_min_alpha},
                }},
                "l1"     : {{}},
                "l2"     : {{
                    'Wv': {l2_wv},
                    'mu': {l2_mu},
                }},
                "sp_weight": {{
                    "h": {sp_weight_h},
                }},
                "sp_targ"  : {{
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
               "batches_per_iter" : 1000,
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
