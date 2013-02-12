#!/bin/sh
DB="postgres://desjagui:cr0quet@opter.iro.umontreal.ca/desjagui_db/icml13_mnistbackimg_ssrbm3_3"

DSET="variation=background_images"
INF="neg_steps=1"
SVM="output_type='h'"
FLAGS="truncate_v=1"
FLAGS="$FLAGS split_norm=0"
HYPER="batch_size=32"
HYPER="$HYPER iscale_wv=0.001 iscale_lambd=0.37 iscale_mu=0.01"
HYPER="$HYPER iscale_hbias=0. iscale_alpha=0.37"
HYPER="$HYPER truncation_bound_v=10.0"
HYPER="$HYPER clip_min_lambd=-10. clip_min_alpha=-10"
HYPER="$HYPER save_every=100000 max_updates=1000000"
HYPER="$HYPER l2_wv=0 l2_mu=0"
HYPER="$HYPER lr_start=0.001 lr_end=0.001"
HYPER="$HYPER standardize_pixels=0"
HYPER="$HYPER gcn=1"
HYPER="$HYPER center=1"
HYPER="$HYPER wv_norm='max_unit'"

# 3 * 2 * 2 = 12
for n_h in 500 1000 2000
do
    for scalar_lambd in 0 1
    do
        for lambd_interaction in 0 1
        do
            PARAMS="$DSET $INF $SVM $FLAGS $HYPER"
            PARAMS="$PARAMS n_h=$n_h bw_s=1 n_v=784"
            PARAMS="$PARAMS scalar_lambd=$scalar_lambd ml_lambd=1"
            PARAMS="$PARAMS lambd_interaction=$lambd_interaction"
            PARAMS="$PARAMS sp_weight_h=0 sp_targ_h=0.1"
            jobman sqlschedule $DB hossrbm.scripts.jobman.ssrbm_mnist_variations_experiment3.experiment $PARAMS
        done
    done
done

# 3 * 2 * 2 = 12
for n_h in 500 1000 2000
do
    for scalar_lambd in 1
    do
        for lambd_interaction in 0
        do
            for sp_weight_h in 1.0 0.1
            do
                for sp_targ_h in 0.1 0.01
                do
                    PARAMS="$DSET $INF $SVM $FLAGS $HYPER"
                    PARAMS="$PARAMS n_h=$n_h bw_s=1 n_v=784"
                    PARAMS="$PARAMS scalar_lambd=$scalar_lambd ml_lambd=0"
                    PARAMS="$PARAMS lambd_interaction=$lambd_interaction"
                    PARAMS="$PARAMS sp_weight_h=$sp_weight_h sp_targ_h=$sp_targ_h"
                    jobman sqlschedule $DB hossrbm.scripts.jobman.ssrbm_mnist_variations_experiment3.experiment $PARAMS
                done
            done
        done
    done
done
