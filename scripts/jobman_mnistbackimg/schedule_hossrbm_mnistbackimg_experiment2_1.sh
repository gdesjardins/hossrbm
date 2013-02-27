#!/bin/sh
DB="postgres://desjagui:cr0quet@opter.iro.umontreal.ca/desjagui_db/icml13_hossrbm_mnistbackimg2_1"

DSET="variation=background_images"
ARCH="sparse_gmask_type=unfactored_g sparse_hmask_type=unfactored_h"
INF="pos_steps=20 neg_steps=2"
SVM="output_type='g+h'"
FLAGS="truncate_s=0 truncate_v=1 lambd_interaction=0 scalar_lambd=1"
FLAGS="$FLAGS wg_norm='none' wh_norm='none' wv_norm='max_mean'"
FLAGS="$FLAGS split_norm=0 mean_field=1 ml_lambd=0 init_mf_rand=1"
HYPER="batch_size=32 iscale_wg=1.0 iscale_wh=1.0"
HYPER="$HYPER iscale_wv=0.01 iscale_lambd=0.37 iscale_mu=0.01"
HYPER="$HYPER iscale_gbias=0. iscale_hbias=0. iscale_vbias=0."
HYPER="$HYPER truncation_bound_v=4.0 truncation_bound_s=3.0"
HYPER="$HYPER clip_min_lambd=-10. clip_min_alpha=-10"
HYPER="$HYPER save_every=50000 max_updates=1000000"
HYPER="$HYPER center=1 gcn=1 standardize_pixels=0"
HYPER="$HYPER iscale_alpha=0."
HYPER="$HYPER lr_start=0.001 lr_end=0.001"
HYPER="$HYPER sp_weight_g=0. sp_weight_h=0."
HYPER="$HYPER sp_targ_g=0.2 sp_targ_h=0.2"

ng=$1
nh=$2
ns=$3
bwg=$4
bwh=$5

PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"
PARAMS="$PARAMS n_g=$ng n_h=$nh n_s=$ns n_v=784 bw_g=$bwg bw_h=$bwh"
jobman sqlschedule $DB hossrbm.scripts.jobman_mnistbackimg.hossrbm_mnistbackimg_experiment2.experiment $PARAMS
