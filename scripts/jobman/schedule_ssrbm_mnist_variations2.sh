#!/bin/sh
DB="postgres://desjagui:cr0quet@opter.iro.umontreal.ca/desjagui_db/icml13_mnistbackimg_ssrbm2"

DSET="variation=background_images"
INF="neg_steps=1"
SVM="output_type='h'"
FLAGS="truncate_v=1 lambd_interaction=0 scalar_lambd=0 wv_norm=max_unit "
FLAGS="$FLAGS split_norm=0 ml_lambd=0"
HYPER="batch_size=32"
HYPER="$HYPER iscale_wv=0.001 iscale_lambd=0.37 iscale_mu=0.01"
HYPER="$HYPER iscale_hbias=0. iscale_alpha=1."
HYPER="$HYPER truncation_bound_v=10.0"
HYPER="$HYPER clip_min_lambd=-10. clip_min_alpha=-10"
HYPER="$HYPER save_every=100000 max_updates=2000000"
HYPER="$HYPER l2_wv=0 l2_mu=0"

for n_h in 300 500 1000 2000
do
    for sp_weight in 0. 0.1 1.0
    do
        for sp_targ in 0.1 0.01
        do
            for lr_start in 0.001
            do
                for lr_end in 0.001 0.0001
                do
                    PARAMS="$DSET $INF $SVM $FLAGS $HYPER"
                    PARAMS="$PARAMS n_h=$n_h bw_s=1 n_v=784"
                    PARAMS="$PARAMS sp_weight_h=$sp_weight"
                    PARAMS="$PARAMS sp_targ_h=$sp_targ"
                    PARAMS="$PARAMS wv_norm=$wv_norm"
                    PARAMS="$PARAMS lr_start=$lr_start lr_end=$lr_end"
                    jobman sqlschedule $DB hossrbm.scripts.jobman.ssrbm_mnist_variations_experiment2.experiment $PARAMS
                done
            done
        done
    done
done
