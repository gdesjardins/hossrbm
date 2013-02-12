#!/bin/sh
DB="postgres://desjagui:cr0quet@opter.iro.umontreal.ca/desjagui_db/icml13_mnistbackimg_ssrbm2_2"

DSET="variation=background_images"
INF="neg_steps=1"
SVM="output_type='h'"
FLAGS="truncate_v=1 lambd_interaction=0 scalar_lambd=0"
FLAGS="$FLAGS split_norm=0 ml_lambd=0"
HYPER="batch_size=32"
HYPER="$HYPER iscale_wv=0.001 iscale_lambd=0.37 iscale_mu=0.01"
HYPER="$HYPER iscale_hbias=0. iscale_alpha=1."
HYPER="$HYPER truncation_bound_v=10.0"
HYPER="$HYPER clip_min_lambd=-10. clip_min_alpha=-10"
HYPER="$HYPER save_every=100000 max_updates=2000000"
HYPER="$HYPER sp_weight_h=0. sp_targ_h=0.1"
HYPER="$HYPER lr_start=0.001 lr_end=0.001"

for n_h in 500 1000 2000
do
    for wv_norm in 'max_unit'
    do
        for l2_mu in 0. 0.1 0.01 0.001
        do
            PARAMS="$DSET $INF $SVM $FLAGS $HYPER"
            PARAMS="$PARAMS n_h=$n_h bw_s=1 n_v=784"
            PARAMS="$PARAMS wv_norm=$wv_norm"
            PARAMS="$PARAMS l2_wv=0 l2_mu=$l2_mu"
            jobman sqlschedule $DB hossrbm.scripts.jobman.ssrbm_mnist_variations_experiment2.experiment $PARAMS
        done
    done
done

for n_h in 500 1000 2000
do
    for wv_norm in 'none'
    do
        for l2_wv in 0.1 0.01 0.001
        do
            PARAMS="$DSET $INF $SVM $FLAGS $HYPER"
            PARAMS="$PARAMS n_h=$n_h bw_s=1 n_v=784"
            PARAMS="$PARAMS wv_norm=$wv_norm"
            PARAMS="$PARAMS l2_wv=$l2_wv l2_mu=0"
            jobman sqlschedule $DB hossrbm.scripts.jobman.ssrbm_mnist_variations_experiment2.experiment $PARAMS
        done
    done
done
