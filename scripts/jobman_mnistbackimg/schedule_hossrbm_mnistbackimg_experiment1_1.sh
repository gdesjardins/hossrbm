#!/bin/sh
DB="postgres://desjagui:cr0quet@opter.iro.umontreal.ca/desjagui_db/icml13_hossrbm_mnistbackimg1_1"

DSET="variation=background_images"
ARCH="sparse_gmask_type=unfactored_g sparse_hmask_type=unfactored_h"
INF="pos_steps=100 neg_steps=2"
SVM="output_type='g+h'"
FLAGS="truncate_s=0 truncate_v=1 lambd_interaction=0 scalar_lambd=1"
FLAGS="$FLAGS wg_norm='none' wh_norm='none' wv_norm='max_unit'"
FLAGS="$FLAGS split_norm=0 mean_field=1 ml_lambd=0"
HYPER="batch_size=32 iscale_wg=1.0 iscale_wh=1.0"
HYPER="$HYPER iscale_wv=0.001 iscale_lambd=0.37 iscale_mu=0.01"
HYPER="$HYPER iscale_gbias=0. iscale_hbias=0. iscale_vbias=0."
HYPER="$HYPER truncation_bound_v=4.0 truncation_bound_s=3.0"
HYPER="$HYPER clip_min_lambd=-10. clip_min_alpha=-10"
HYPER="$HYPER save_every=50000 max_updates=1000000"
HYPER="$HYPER center=1 gcn=1 standardize_pixels=0"
HYPER="$HYPER iscale_alpha=0."
HYPER="$HYPER lr_start=0.001 lr_end=0.001"
HYPER="$HYPER sp_targ_g=0.2 sp_targ_h=0.2"

for sp_weight in 0. 0.2
do
    HYPER="$HYPER sp_weight_g=$sp_weight sp_weight_h=$sp_weight"

    ######## FULLY CONNECTED ###################

    PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
    PARAMS="$PARAMS n_g=55 n_h=55 n_s=3025 n_v=784 bw_g=55 bw_h=55"
    jobman sqlschedule $DB hossrbm.scripts.jobman_mnistbackimg.hossrbm_mnistbackimg_experiment1.experiment $PARAMS

    PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
    PARAMS="$PARAMS n_g=38 n_h=80 n_s=3040 n_v=784 bw_g=38 bw_h=80"
    jobman sqlschedule $DB hossrbm.scripts.jobman_mnistbackimg.hossrbm_mnistbackimg_experiment1.experiment $PARAMS

    PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
    PARAMS="$PARAMS n_g=24 n_h=125 n_s=3000 n_v=784 bw_g=24 bw_h=125"
    jobman sqlschedule $DB hossrbm.scripts.jobman_mnistbackimg.hossrbm_mnistbackimg_experiment1.experiment $PARAMS

    PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
    PARAMS="$PARAMS n_g=10 n_h=300 n_s=3000 n_v=784 bw_g=10 bw_h=300"
    jobman sqlschedule $DB hossrbm.scripts.jobman_mnistbackimg.hossrbm_mnistbackimg_experiment1.experiment $PARAMS

    ######## 20 BLOCKS ################

    PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
    PARAMS="$PARAMS n_g=240 n_h=240 n_s=2880 n_v=784 bw_g=12 bw_h=12"
    jobman sqlschedule $DB hossrbm.scripts.jobman_mnistbackimg.hossrbm_mnistbackimg_experiment1.experiment $PARAMS

    PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
    PARAMS="$PARAMS n_g=160 n_h=360 n_s=2880 n_v=784 bw_g=8 bw_h=18"
    jobman sqlschedule $DB hossrbm.scripts.jobman_mnistbackimg.hossrbm_mnistbackimg_experiment1.experiment $PARAMS

    PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
    PARAMS="$PARAMS n_g=120 n_h=500 n_s=3000 n_v=784 bw_g=6 bw_h=25"
    jobman sqlschedule $DB hossrbm.scripts.jobman_mnistbackimg.hossrbm_mnistbackimg_experiment1.experiment $PARAMS

    PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
    PARAMS="$PARAMS n_g=80 n_h=720 n_s=2880 n_v=784 bw_g=4 bw_h=36"
    jobman sqlschedule $DB hossrbm.scripts.jobman_mnistbackimg.hossrbm_mnistbackimg_experiment1.experiment $PARAMS

    ######## 100 BLOCKS ################

    PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
    PARAMS="$PARAMS n_g=500 n_h=600 n_s=3000 n_v=784 bw_g=5 bw_h=6"
    jobman sqlschedule $DB hossrbm.scripts.jobman_mnistbackimg.hossrbm_mnistbackimg_experiment1.experiment $PARAMS

    PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
    PARAMS="$PARAMS n_g=400 n_h=800 n_s=3200 n_v=784 bw_g=4 bw_h=8"
    jobman sqlschedule $DB hossrbm.scripts.jobman_mnistbackimg.hossrbm_mnistbackimg_experiment1.experiment $PARAMS

    PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
    PARAMS="$PARAMS n_g=300 n_h=1500 n_s=3000 n_v=784 bw_g=2 bw_h=10"
    jobman sqlschedule $DB hossrbm.scripts.jobman_mnistbackimg.hossrbm_mnistbackimg_experiment1.experiment $PARAMS


    ######## 500 BLOCKS ################

    PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
    PARAMS="$PARAMS n_g=1500 n_h=1500 n_s=3000 n_v=784 bw_g=2 bw_h=2"
    jobman sqlschedule $DB hossrbm.scripts.jobman_mnistbackimg.hossrbm_mnistbackimg_experiment1.experiment $PARAMS

    PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
    PARAMS="$PARAMS n_g=1000 n_h=1500 n_s=3000 n_v=784 bw_g=2 bw_h=3"
    jobman sqlschedule $DB hossrbm.scripts.jobman_mnistbackimg.hossrbm_mnistbackimg_experiment1.experiment $PARAMS

    PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
    PARAMS="$PARAMS n_g=999 n_h=999 n_s=2997 n_v=784 bw_g=3 bw_h=3"
    jobman sqlschedule $DB hossrbm.scripts.jobman_mnistbackimg.hossrbm_mnistbackimg_experiment1.experiment $PARAMS

done
