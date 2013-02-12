#!/bin/sh
DB="postgres://desjagui:cr0quet@opter.iro.umontreal.ca/desjagui_db/icml13_mnistbackimg_ssrbm3"

DSET="variation=background_images"
INF="neg_steps=1"
SVM="output_type='h'"
FLAGS="truncate_v=1 lambd_interaction=0 wv_norm=max_unit "
FLAGS="$FLAGS split_norm=0 ml_lambd=0"
HYPER="batch_size=32"
HYPER="$HYPER iscale_wv=0.001 iscale_lambd=0.37 iscale_mu=0.01"
HYPER="$HYPER iscale_hbias=0. iscale_alpha=0.37"
HYPER="$HYPER truncation_bound_v=10.0"
HYPER="$HYPER clip_min_lambd=-10. clip_min_alpha=-10"
HYPER="$HYPER save_every=100000 max_updates=2000000"
HYPER="$HYPER l2_wv=0 l2_mu=0"
HYPER="$HYPER sp_weight_h=0 sp_targ_h=0.1"
HYPER="$HYPER lr_start=0.001 lr_end=0.001"

for n_h in 1000 2000
do
    for center in 0 1
    do
        for gcn in 0 1
        do
            for standardize_pixels in 0 1
            do
                for scalar_lambd in 0 1
                do
                    PARAMS="$DSET $INF $SVM $FLAGS $HYPER"
                    PARAMS="$PARAMS n_h=$n_h bw_s=1 n_v=784"
                    PARAMS="$PARAMS center=$center"
                    PARAMS="$PARAMS gcn=$gcn"
                    PARAMS="$PARAMS standardize_pixels=$standardize_pixels"
                    PARAMS="$PARAMS scalar_lambd=$scalar_lambd"
                    jobman sqlschedule $DB hossrbm.scripts.jobman.ssrbm_mnist_variations_experiment3.experiment $PARAMS
                done
            done
        done
    done
done
