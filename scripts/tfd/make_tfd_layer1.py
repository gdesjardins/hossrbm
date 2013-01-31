from pylearn2.utils import serial
from pylearn2.utils import string_utils
import numpy
import argparse

from hossrbm.scripts.conv_pipeline import cssrbm_feature_extractor as featext

print "Preparing output directory..."
data_dir = string_utils.preprocess('/data/lisatmp2/desjagui/data')
indir  = data_dir + '/tfd_cn'
outdir = data_dir + '/tfd_cn_layer2'
serial.mkdir(outdir)

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path of model .pkl file.')
args = parser.parse_args()

"""
print 'Processing unlabeled set...'
in_dset_fname = '%s/%s.pkl' % (indir, 'unlabeled')
out_dset_fname = '%s/%s.pkl' % (outdir, 'unlabeled')
featext.run(args.model,
        in_dset_fname,
        batch_size = 128,
        image_width = 48,
        patch_width = 14,
        pool_width = 12,
        output_width = 9216,
        output_file = out_dset_fname)
"""

for fold_i in xrange(0, 4):

    print 'Processing fold %i...' % fold_i

    indir_fold  = '%s/fold%i' % (indir, fold_i)
    outdir_fold = '%s/fold%i' % (outdir, fold_i)
    serial.mkdir(outdir_fold)

    for dset in ['train', 'valid', 'test']:

        in_dset_fname = '%s/%s.pkl' % (indir_fold, dset)
        out_dset_fname = '%s/%s.pkl' % (outdir_fold, dset)

        featext.run(args.model,
                in_dset_fname,
                batch_size = 128,
                image_width = 48,
                patch_width = 14,
                pool_width = 12,
                output_width = 9216,
                output_file = out_dset_fname)
