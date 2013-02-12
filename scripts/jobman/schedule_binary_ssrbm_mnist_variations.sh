#!/bin/sh
DB="postgres://desjagui:cr0quet@opter.iro.umontreal.ca/desjagui_db/icml13_mnistbackimg_binssrbm"

DSET="variation=background_images"
INF="neg_steps=1"
SVM="output_type='h'"
FLAGS="split_norm=0"
HYPER="batch_size=32"
HYPER="$HYPER iscale_wv=0.001 iscale_mu=0.01"
HYPER="$HYPER iscale_hbias=0. iscale_vbias=0."
HYPER="$HYPER iscale_alpha=-2. clip_min_alpha=-10"
HYPER="$HYPER save_every=100000 max_updates=2000000"
HYPER="$HYPER lr_start=0.001 lr_end=0.001"
HYPER="$HYPER l2_mu=0. l2_wv=0."

for n_h in 500 1000 2000
do
    for wv_norm in 'none' 'max_unit'
    do
        for sp_weight_h in 0.
        do
            PARAMS="$DSET $INF $SVM $FLAGS $HYPER"
            PARAMS="$PARAMS n_h=$n_h bw_s=1 n_v=784"
            PARAMS="$PARAMS wv_norm=$wv_norm"
            PARAMS="$PARAMS sp_weight_h=$sp_weight_h sp_targ_h=0"
            jobman sqlschedule $DB hossrbm.scripts.jobman.binary_ssrbm_mnist_variations_experiment.experiment $PARAMS
        done
    done
done

for n_h in 500 1000 2000
do
    for wv_norm in 'none' 'max_unit'
    do
        for sp_weight_h in 0.1 1.0
        do
            for sp_targ_h in 0.1 0.01
            do
                PARAMS="$DSET $INF $SVM $FLAGS $HYPER"
                PARAMS="$PARAMS n_h=$n_h bw_s=1 n_v=784"
                PARAMS="$PARAMS wv_norm=$wv_norm"
                PARAMS="$PARAMS sp_weight_h=$sp_weight_h sp_targ_h=$sp_targ_h"
                jobman sqlschedule $DB hossrbm.scripts.jobman.binary_ssrbm_mnist_variations_experiment.experiment $PARAMS
            done
        done
    done
done
