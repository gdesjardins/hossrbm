#!/bin/sh
DB="postgres://desjagui:cr0quet@opter.iro.umontreal.ca/desjagui_db/icml13_tfdlcn_hossrbm"

DSET="variation=background_images"
ARCH="sparse_gmask_type=unfactored_g sparse_hmask_type=unfactored_h"
INF="pos_steps=100 neg_steps=10"
SVM="output_type='g+h'"
FLAGS="truncate_s=0 truncate_v=1 lambd_interaction=0 scalar_lambd=0"
FLAGS="$FLAGS wg_norm='none' wh_norm='none' split_norm=0 mean_field=1"
HYPER="batch_size=32 iscale_wg=1.0 iscale_wh=1.0"
HYPER="$HYPER iscale_wv=0.01 iscale_lambd=0.37 iscale_mu=0.1 iscale_alpha=-2.0"
HYPER="$HYPER iscale_gbias=0.8 iscale_hbias=0.8 iscale_vbias=0."
HYPER="$HYPER truncation_bound_v=2.0 truncation_bound_s=3.0"
HYPER="$HYPER clip_min_lambd=-10. clip_min_alpha=-10"
HYPER="$HYPER sp_targ_g=0.2 sp_targ_h=0.2"
HYPER="$HYPER save_every=100000 max_updates=1000000"


for n_g in 333 500 1000
do
    for sp_weight in 0 0.1
    do
        for wv_norm in ['none','max_unit']
        do
            for lr in 0.001 0.0001
            do
                PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
                PARAMS="$PARAMS n_g=$n_g n_h=$n_g n_s=900 n_v=784 bw_g=3 bw_h=3"
                PARAMS="$PARAMS sp_weight_g=$sp_weight sp_weight_h=$sp_weight"
                PARAMS="$PARAMS lr_start=$lr lr_end=$lr"
                PARAMS="$PARAMS wv_norm=$wv_norm"
                jobman sqlschedule $DB hossrbm.scripts.jobman.experiment_hossrbm.experiment $PARAMS
            done
        done

    done
done

