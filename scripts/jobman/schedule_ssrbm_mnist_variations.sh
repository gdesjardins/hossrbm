#!/bin/sh
DB="postgres://desjagui:cr0quet@opter.iro.umontreal.ca/desjagui_db/icml13_mnistbackimg_ssrbm"

DSET="variation=background_images"
INF="neg_steps=1"
SVM="output_type='h'"
FLAGS="truncate_v=1 lambd_interaction=0 scalar_lambd=0 wv_norm=max_unit "
FLAGS="$FLAGS split_norm=0 ml_lambd=0"
HYPER="batch_size=32"
HYPER="$HYPER iscale_wv=0.01 iscale_lambd=0.37 iscale_mu=0.1"
HYPER="$HYPER iscale_hbias=0. iscale_alpha=-2."
HYPER="$HYPER truncation_bound_v=2.0"
HYPER="$HYPER clip_min_lambd=-10. clip_min_alpha=-10"
HYPER="$HYPER sp_targ_h=0.1"
HYPER="$HYPER save_every=100000 max_updates=1000000"


for n_h in 500 1000 2000
do
    for sp_weight in 0. 0.1
    do
        for lr in 0.001
        do
            PARAMS="$DSET $INF $SVM $FLAGS $HYPER"
            PARAMS="$PARAMS n_h=$n_h bw_s=1 n_v=784"
            PARAMS="$PARAMS sp_weight_g=$sp_weight sp_weight_h=$sp_weight"
            PARAMS="$PARAMS wv_norm=$wv_norm"
            PARAMS="$PARAMS lr_start=$lr lr_end=$lr"
            jobman sqlschedule $DB hossrbm.scripts.jobman.ssrbm_mnist_variations_experiment.experiment $PARAMS
        done
    done
done

