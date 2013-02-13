#!/bin/sh
DB="postgres://desjagui:cr0quet@opter.iro.umontreal.ca/desjagui_db/icml13_tfdlayer2_hossrbm1_1"

DSET="fold=0"
ARCH="sparse_gmask_type=unfactored_g sparse_hmask_type=unfactored_h"
INF="pos_steps=100 neg_steps=10"
SVM="output_type='g+h'"
FLAGS="wg_norm='none' wh_norm='none' split_norm=0 mean_field=1"
HYPER="batch_size=32 iscale_wg=1.0 iscale_wh=1.0"
HYPER="$HYPER iscale_wv=0.001 iscale_mu=0.01"
HYPER="$HYPER iscale_gbias=0.8 iscale_hbias=0.8 iscale_vbias=0."
HYPER="$HYPER clip_min_alpha=-10"
HYPER="$HYPER sp_targ_g=0. sp_targ_h=0."
HYPER="$HYPER sp_weight_g=0. sp_weight_h=0."
HYPER="$HYPER save_every=100000 max_updates=1000000"
HYPER="$HYPER wv_norm='max_unit'"
HYPER="$HYPER lr_start=0.001 lr_end=0.001"

for iscale_alpha in -2 0.37
do
    for lr in 0.001 0.0001
    do
        # 1500 filters, 2x2 pooling: ng=ng=750
        # 3000 filters, 2x2 pooling: ng=ng=1500
        for n_g in 750 1500
        do
            block_size=2
            n_s=`expr $n_g "*" $block_size`
            PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
            PARAMS="$PARAMS n_g=$n_g n_h=$n_g n_s=$n_s n_v=784 bw_g=$block_size bw_h=$block_size"
            PARAMS="$PARAMS lr_start=$lr lr_end=$lr"
            PARAMS="$PARAMS iscale_alpha=$iscale_alpha"
            jobman sqlschedule $DB hossrbm.scripts.jobman_tfd.tfdlayer2_hossrbm1_experiment1.experiment $PARAMS
        done

        # 1500 filters, 3x3 pooling: ng=ng=500
        # 3000 filters, 3x3 pooling: ng=ng=1000
        for n_g in 501 999
        do
            block_size=3
            n_s=`expr $n_g "*" $block_size`
            PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
            PARAMS="$PARAMS n_g=$n_g n_h=$n_g n_s=$n_s n_v=784 bw_g=$block_size bw_h=$block_size"
            PARAMS="$PARAMS lr_start=$lr lr_end=$lr"
            PARAMS="$PARAMS iscale_alpha=$iscale_alpha"
            jobman sqlschedule $DB hossrbm.scripts.jobman_tfd.tfdlayer2_hossrbm1_experiment1.experiment $PARAMS
        done

        # 1500 filters, 5x5 pooling: ng=ng=300
        # 3000 filters, 5x5 pooling: ng=ng=600
        #for n_g in 300 6000
        #do
            #block_size=5
            #n_s=`expr $n_g "*" $block_size`
            #PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
            #PARAMS="$PARAMS n_g=$n_g n_h=$n_g n_s=$n_s n_v=784 bw_g=$block_size bw_h=$block_size"
            #PARAMS="$PARAMS lr_start=$lr lr_end=$lr"
            #PARAMS="$PARAMS iscale_alpha=$iscale_alpha"
            #jobman sqlschedule $DB hossrbm.scripts.jobman_tfd.tfdlayer2_hossrbm1_experiment1.experiment $PARAMS
        #done
    done
done
