"""
This script makes a dataset of two million approximately whitened patches, extracted at random uniformly
from the CIFAR-100 train dataset.

This script is intended to reproduce the preprocessing used by Adam Coates et. al. in their work from
the first half of 2011 on the CIFAR-10 and STL-10 datasets.
"""

from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.datasets.tfd import TFD
from pylearn2.utils import string_utils
from hossrbm import preproc as my_preproc

data_dir = string_utils.preprocess('/data/lisatmp2/desjagui/data')

pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.GlobalContrastNormalization(subtract_mean=True))
pipeline.items.append(my_preproc.LeCunLCN((1,48,48)))
pipeline.items.append(preprocessing.RemoveMean(axis=0))
pipeline.items.append(preprocessing.ExtractPatches(patch_shape=(14,14), num_patches=5*1000*1000))

#### Build full-sized image dataset. ####
print "Preparing output directory for unlabeled patches..."
outdir = data_dir + '/tfd_lcn_v1'
serial.mkdir(outdir)
README = open('README','w')
README.write("""
File generated from hossrbm/scripts/tfd/make_tfd_lcn.py.
""")
README.close()

print 'Loading TFD unlabeled dataset...'
print "Preprocessing the data..."
data = TFD('unlabeled')
data.apply_preprocessor(preprocessor = pipeline, can_fit = True)
data.use_design_loc(outdir + '/unlabeled_patches.npy')
serial.save(outdir + '/unlabeled_patches.pkl',data)

#### For supervised dataset, we work on the full-image dataset ####
pipeline.items.pop()

#### Build supervised-training datasets ####
print "Preparing output directory for supervised data..."
for fold_i in xrange(0,5):

    path = '%s/fold%i' % (outdir, fold_i)
    serial.mkdir(path)

    train_data = TFD('train', fold=fold_i, center=False, shuffle=True, seed=37192)
    train_data.apply_preprocessor(preprocessor = pipeline, can_fit = False)
    print 'Saving train dataset...'
    serial.save(path + '/train.pkl', train_data)

    valid_data = TFD('valid', fold=fold_i, center=False, shuffle=True, seed=37192)
    valid_data.apply_preprocessor(preprocessor = pipeline, can_fit = False)
    print 'Saving valid dataset...'
    serial.save(path + '/valid.pkl', valid_data)

    test_data  = TFD('test',  fold=fold_i, center=False, shuffle=True, seed=37192)
    test_data.apply_preprocessor(preprocessor = pipeline, can_fit = False)
    print 'Saving test dataset...'
    serial.save(path + '/test.pkl', test_data)
    
serial.save(outdir + '/pipeline.pkl', pipeline)
