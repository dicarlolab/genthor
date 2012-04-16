import os
import itertools
import re
import cPickle

import numpy as np
import Image
import boto
import tabular as tb
from yamutils.fast import reorder_to, isin

import skdata.larray as larray
from skdata.data_home import get_data_home
from skdata.utils.download_and_extract import download_boto, extract, download


class TrainingDataset(object):

    FILES = [('genthor_training_data_20120416.zip',
              'cc5cbb5fd25cb469783e2494d7efdf1d189035a5')]

    name = 'GenthorTrainingDataset'
    
    def __init__(self):
        pass

    def home(self, *suffix_paths):
        return os.path.join(get_data_home(), self.name, *suffix_paths)

    def fetch(self):
        """Download and extract the dataset."""
        home = self.home()
        if not os.path.exists(home):
            os.makedirs(home)
        for base, sha1 in self.FILES:
            archive_filename = os.path.join(home, base)  
            if not os.path.exists(archive_filename):
                url = 'http://dicarlocox-datasets.s3.amazonaws.com/' + base
                print ('downloading %s' % url)
                download(url, archive_filename, sha1=sha1, verbose=True)                
                extract(archive_filename, home, sha1=sha1, verbose=True)
                           
    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self.fetch()
            self._meta = self._get_meta()
        return self._meta
        
    def _get_meta(self): 
        homedir = os.path.join(self.home(), 'genthor_training_data_20120412')
        L = sorted(os.listdir(homedir))
        imgs = filter(lambda x: x.endswith('.jpg'), L)
        pkls = filter(lambda x: x.endswith('.pkl'), L)
        assert len(imgs) == len(pkls)
        recs = []
        for imfile, pklfile in zip(imgs, pkls):
            pkl = cPickle.load(open(os.path.join(homedir, pklfile)))
            file_id = os.path.split(imfile)[-1]
            rec = (os.path.join(homedir, imfile), 
                   pkl['bgname'],
                   pkl['bghp'][0],
                   pkl['bghp'][1],
                   pkl['bgscale'][0],
                   pkl['category'],
                   pkl['modelname'],
                   pkl['hpr'][0],
                   pkl['hpr'][1],
                   pkl['hpr'][2],
                   pkl['pos'][0],
                   pkl['pos'][1],
                   pkl['scale'][0],
                   file_id)
            recs.append(rec)
        meta = tb.tabarray(records=recs, names=['filename',
                                             'bgname',
                                             'bgh',
                                             'bgp',
                                             'bgscale',
                                             'category',
                                             'model_id',
                                             'ryz',
                                             'rxz',
                                             'rxy',
                                             'ty',
                                             'tz',
                                             's',
                                             'file_id'])
        return meta
        
    @property 
    def filenames(self):
        return self.meta['filename']

    def get_images(self, dtype, preproc):
        self.fetch()
        size = tuple(preproc['size'])
        normalize = preproc['global_normalize']
        mode = preproc['mode']
        return larray.lmap(ImgLoaderResizer(inshape=(256, 256),
                                            shape=size,
                                            dtype=dtype,
                                            normalize=normalize,
                                            mode=mode),
                                self.filenames)

    ####TODO:  split-generating methods
    def get_splits(self, ntrain, ntest, nvalidate, num_splits):
        pass
        

#TODO: test splits
def test_training_dataset():
    dataset = TrainingDataset()
    meta = dataset.meta
    assert len(meta) == 11000
    agg = meta[['model_id', 'category']].aggregate(['category'],
                                             AggFunc=lambda x: len(x))
    assert agg.tolist() == [('boats', 1000),
                             ('buildings', 1000),
                             ('cars', 1000),
                             ('cats_and_dogs', 1000),
                             ('chair', 1000),
                             ('faces', 1000),
                             ('guns', 1000),
                             ('planes', 1000),
                             ('plants', 1000),
                             ('reptiles', 1000),
                             ('table', 1000)]

    agg2 = meta[['model_id', 'category']].aggregate(['category'], 
           AggFunc=lambda x : len(np.unique(x)))       
    assert agg2.tolist() == [('boats', 10),
         ('buildings', 10),
         ('cars', 10),
         ('cats_and_dogs', 10),
         ('chair', 10),
         ('faces', 10),
         ('guns', 10),
         ('planes', 10),
         ('plants', 10),
         ('reptiles', 10),
         ('table', 10)]
         
    imgs = dataset.get_images('float32', {'size':(256, 256),
                          'global_normalize':False, 'mode':'L'})
    assert imgs.shape == (11000, 256, 256)


class ImgLoaderResizer(object):
    """
    """
    def __init__(self,
                 inshape, 
                 shape=None,
                 ndim=None,
                 dtype='float32',
                 normalize=True,
                 mode='RGB',
                 crop=None,
                 mask=None):
        self.inshape = inshape
        assert len(shape) == 2
        shape = tuple(shape)
        if crop is None:
            crop = (0, 0, self.inshape[0], self.inshape[1])
        assert len(crop) == 4
        crop = tuple(crop)
        l, t, r, b = crop
        assert 0 <= l < r <= self.inshape[0]
        assert 0 <= t < b <= self.inshape[1]
        self._crop = crop   
        assert dtype == 'float32'
        self.dtype=dtype
        self._shape = shape
        if ndim is None:
            self._ndim = None if (shape is None) else len(shape)
        else:
            self._ndim = ndim
        self._dtype = dtype
        self.normalize = normalize
        self.mask=mask
        ##XXX:  To-do allow for other image modes (e.g. RGB)
        assert mode == 'L'
        self.mode=mode

    def rval_getattr(self, attr, objs):
        if attr == 'shape' and self._shape is not None:
            return self._shape
        if attr == 'ndim' and self._ndim is not None:
            return self._ndim
        if attr == 'dtype':
            return self._dtype
        raise AttributeError(attr)

    def __call__(self, file_path):
        im = Image.open(file_path)
        if im.mode != self.mode:
            im = im.convert(self.mode)
        assert im.size == self.inshape
        if self.mask is not None:
            im.paste(self.mask)
        if self._crop != (0, 0,) + self.inshape:
            im = im.crop(self._crop)
        l, t, r, b = self._crop
        assert im.size == (r - l, b - t)
        if max(im.size) != self._shape[0]:
            m = self._shape[0]/float(max(im.size))
            new_shape = (int(round(im.size[0]*m)), int(round(im.size[1]*m)))
            im = im.resize(new_shape, Image.ANTIALIAS)
        imval = np.asarray(im, 'float32')
        rval = np.zeros(self._shape, dtype=self.dtype)
        ctr = self._shape[0]/2
        cxmin = ctr - imval.shape[0] / 2
        cxmax = ctr - imval.shape[0] / 2 + imval.shape[0]
        cymin = ctr - imval.shape[1] / 2
        cymax = ctr - imval.shape[1] / 2 + imval.shape[1]
        rval[cxmin:cxmax,cymin:cymax] = imval
        if self.normalize:
            rval -= rval.mean()
            rval /= max(rval.std(), 1e-3)
        else:
            rval /= 255.0
        assert rval.shape == self._shape
        return rval
    
    
