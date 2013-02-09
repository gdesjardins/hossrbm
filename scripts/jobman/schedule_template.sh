#!/bin/sh
DB="postgres://desjagui:cr0quet@opter.iro.umontreal.ca/desjagui_db/icml13_tfdlcn_hossrbm"

DSET="variation=background_images"
ARCH="n_g=300 n_h=300 n_s=900 n_v=784 bw_g=3 bw_h=3"
ARCH="$ARCH sparse_gmask_type=unfactored_g sparse_hmask_type=unfactored_h"
INF="pos_steps=100 neg_steps=10"
SVM="output_type='h+s'"
FLAGS="truncate_s=0 truncate_v=1 lambd_interaction=0 scalar_lambd=0"
FLAGS="$FLAGS wg_norm='none' wh_norm='none' wv_norm='max_unit' split_norm=0 mean_field=1"
HYPER="batch_size=32 lr_start=0.001 lr_end=0.001 iscale_wg=1.0 iscale_wh=1.0"
HYPER="$HYPER iscale_wv=0.01 iscale_lambd=0.37 iscale_mu=0.1 iscale_alpha=-2.0"
HYPER="$HYPER iscale_gbias=0.8 iscale_hbias=0.8 iscale_vbias=0."
HYPER="$HYPER truncation_bound_v=2.0 truncation_bound_s=3.0"
HYPER="$HYPER clip_min_lambd=-10. clip_min_alpha=-10"
HYPER="$HYPER sp_weight_g=0 sp_weight_h=0"
HYPER="$HYPER sp_targ_g=0 sp_targ_h=0"
HYPER="$HYPER save_every=100000 max_updates=1000000"

PARAMS="$DSET $ARCH $INF $SVM $FLAGS $HYPER"

jobman cmdline hossrbm.scripts.jobman.experiment_hossrbm.experiment $PARAMS

