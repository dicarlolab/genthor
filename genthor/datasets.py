import os
import itertools
import re
import hashlib
import cPickle

import numpy as np
import Image
import tabular as tb
from yamutils.fast import reorder_to, isin
from yamutils.basic import dict_inverse

import pyll
choice = pyll.scope.choice
uniform = pyll.scope.uniform
loguniform = pyll.scope.loguniform
import pyll.stochastic as stochastic

import skdata.larray as larray
from skdata.data_home import get_data_home
from skdata.utils.download_and_extract import extract, download

import genthor.renderer.genthor_renderer as gr
import genthor.model_info as model_info


class DatasetBase(object):
    def home(self, *suffix_paths):
        return os.path.join(get_data_home(), self.base_name, *suffix_paths)

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

    def get_subset_splits(self, *args, **kwargs):
        return get_subset_splits(self.meta, *args, **kwargs) 


class GenerativeDatasetBase(DatasetBase):
    bg_root = 'genthor_backgrounds_20120418'
    model_root = 'genthor_processed_models_20120418'
    FILES = [('genthor_backgrounds_20120418.zip',
              '17c5dd97be0775a7e6ec5de07592a44fb4551b76'),
              ('genthor_processed_models_20120418.tar.gz',
               'fd9a26a2b8198a7745ff642c4351f5206fc7d550'
               )]

    base_name = 'GenthorGenerative'
    
    def _get_meta(self):
        #generate params 
        models = self.models
        templates = self.templates

        latents = []
        rng = np.random.RandomState(seed=0)

        model_categories = dict_inverse(model_info.MODEL_CATEGORIES)
        for tdict in templates:
            n_ex_per_model = tdict['n_ex_per_model']
            template = tdict['template']
            tname = tdict['name']
            for model in models:
                print('Generating meta for %s' % model)
                for _ind in range(n_ex_per_model):
                    l = stochastic.sample(template, rng)
                    l['obj'] = model
                    l['category'] = model_categories[model][0]
                    l['id'] = get_image_id(l)
                    rec = (l['bgname'],
                           float(l['bgphi']),
                           float(l['bgpsi']),
                           float(l['bgscale']),
                           l['category'],
                           l['obj'],
                           float(l['ryz']),
                           float(l['rxz']),
                           float(l['rxy']),
                           float(l['ty']),
                           float(l['tz']),
                           float(l['s']),
                           tname,
                           l['id'])
                    latents.append(rec)
        meta = tb.tabarray(records=latents, names = ['bgname',
                                                     'bgphi',
                                                     'bgpsi',
                                                     'bgscale',
                                                     'category',
                                                     'obj',
                                                     'ryz',
                                                     'rxz',
                                                     'rxy',
                                                     'ty',
                                                     'tz',
                                                     's',
                                                     'tname',
                                                     'id'])
        return meta

    def get_images(self, preproc):
        name = self.specific_name + '_' + get_image_id(preproc)
        basedir = self.home()
        cache_file = os.path.join(basedir, name)
        meta = self.meta
        window_type = 'offscreen'
        size = preproc['size']
        lbase, output = gr.setup_renderer(window_type, size=size) 
        model_root = self.home(self.model_root)
        bg_root = self.home(self.bg_root)
        images = larray.lmap(ImgRendererResizer(model_root, bg_root, preproc,
                                                lbase, output), meta)
        return larray.cache_memmap(images, name=name, basedir=basedir)
        

def get_subset_splits(meta, npc_train, npc_tests, num_splits,
                      catfunc, train_q=None, test_qs=None, test_names=None, 
                      npc_validate=0):
    train_inds = np.arange(len(meta)).astype(np.int)
    if test_qs is None:
        test_qs = [test_qs]
    if test_names is None:
        assert len(test_qs) == 1
        test_names = ['test']
    else:
        assert len(test_names) == len(test_qs)
        assert 'train' not in test_names
    test_ind_list = [np.arange(len(meta)).astype(np.int) \
                                              for _ in range(len(test_qs))]
    if train_q is not None:
        sub = np.array(map(train_q, meta)).astype(np.bool)
        train_inds = train_inds[sub]
    for _ind, test_q in enumerate(test_qs):
        if test_q is not None:
             sub = np.array(map(test_q, meta)).astype(np.bool)
             test_ind_list[_ind] = test_ind_list[_ind][sub]
    
    all_test_inds = list(itertools.chain(*test_ind_list))
    all_inds = np.sort(np.unique(train_inds.tolist() + all_test_inds))
    categories = np.array(map(catfunc, meta))
    ucategories = np.unique(categories[all_inds])    
    rng = np.random.RandomState(0)  #or do you want control over the seed?
    splits = [dict([('train', [])] + \
                   [(tn, []) for tn in test_names]) for _ in range(num_splits)]
    validations = [[] for _ in range(len(test_qs))]
    for cat in ucategories:
        cat_validates = []
        ctils = []
        for _ind, test_inds in enumerate(test_ind_list):
            cat_test_inds = test_inds[categories[test_inds] == cat]
            ctils.append(len(cat_test_inds))
            if npc_validate > 0:
                assert len(cat_test_inds) >= npc_validate, (
                 'not enough to validate')
                pv = rng.permutation(len(cat_test_inds))
                cat_validate = cat_test_inds[pv[:npc_validate]]
                validations[_ind] += cat_validate.tolist()
            else:
                cat_validate = []
            cat_validates.extend(cat_validate)
        cat_validates = np.sort(np.unique(cat_validates))
        for split_ind in range(num_splits):
            cat_train_inds = train_inds[categories[train_inds] == cat]
            if len(cat_train_inds) < np.mean(ctils):    
                cat_train_inds = train_inds[categories[train_inds] == cat]
                cat_train_inds = np.array(
                        list(set(cat_train_inds).difference(cat_validates)))            
                assert len(cat_train_inds) >= npc_train, ( 
                                    'not enough train for %s, %d, %d' % (cat,
                                                len(cat_train_inds), npc_train))
                cat_train_inds.sort()
                p = rng.permutation(len(cat_train_inds))
                cat_train_inds_split = cat_train_inds[p[:npc_train]]
                splits[split_ind]['train'] += cat_train_inds_split.tolist()
                for _ind, test_inds in enumerate(test_ind_list):
                    npc_test = npc_tests[_ind]
                    cat_test_inds = test_inds[categories[test_inds] == cat]
                    cat_test_inds_c = np.array(list(
                             set(cat_test_inds).difference(
                             cat_train_inds_split).difference(cat_validates)))
                    assert len(cat_test_inds_c) >= npc_test, (
                                          'not enough test for %s %d %d' % 
                                      (cat, len(cat_test_inds_c), npc_test))
                    p = rng.permutation(len(cat_test_inds_c))
                    cat_test_inds_split = cat_test_inds_c[p[: npc_test]]
                    name = test_names[_ind]
                    splits[split_ind][name] += cat_test_inds_split.tolist()
            else:
                all_cat_test_inds = []
                for _ind, test_inds in enumerate(test_ind_list):
                    npc_test = npc_tests[_ind]
                    cat_test_inds = test_inds[categories[test_inds] == cat]
                    cat_test_inds_c = np.sort(np.array(list(
                             set(cat_test_inds).difference(cat_validates))))
                    assert len(cat_test_inds_c) >= npc_test, (
                                    'not enough test for %s %d %d' %
                                      (cat, len(cat_test_inds_c), npc_test))
                    p = rng.permutation(len(cat_test_inds_c))
                    cat_test_inds_split = cat_test_inds_c[p[: npc_test]]
                    name = test_names[_ind]
                    splits[split_ind][name] += cat_test_inds_split.tolist()
                    all_cat_test_inds.extend(cat_test_inds_split)
                cat_train_inds = np.array(list(set(cat_train_inds).difference(
                                 all_cat_test_inds).difference(cat_validates)))
                assert len(cat_train_inds) >= npc_train, (
                               'not enough train for %s, %d, %d' % 
                               (cat, len(cat_train_inds), npc_train))
                cat_train_inds.sort()
                p = rng.permutation(len(cat_train_inds))
                cat_train_inds_split = cat_train_inds[p[:npc_train]]
                splits[split_ind]['train'] += cat_train_inds_split.tolist()
            
    return splits, validations


def get_image_id(l):
    return hashlib.sha1(repr(l)).hexdigest()


def get_tmpfilename():
    return 'tmpfile_' + str(np.random.randint(1e8))

    
class GenerativeDataset1(GenerativeDatasetBase):    
    models = model_info.MODEL_SUBSET_1
    bad_backgrounds = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg',
                       'INTERIOR_11ST.jpg', 'INTERIOR_10ST.jpg',
                       'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
                       'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg',
                       'INTERIOR_05ST.jpg']
    good_backgrounds = [_b for _b in model_info.BACKGROUNDS
                                                  if _b not in bad_backgrounds]
    templates = [{'n_ex_per_model': 10,
                  'name': 'var0',  
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': 1,
                     'ty': 0,
                     'tz': 0,
                     'ryz': 0,
                     'rxy': 0,
                     'rxz': 0,
                     }
                 },
                 {'n_ex_per_model': 50,
                  'name': 'translation_scale', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': loguniform(np.log(2./3), np.log(2.)),
                     'ty': uniform(-1.0, 1.0),
                     'tz': uniform(-1.0, 1.0),
                     'ryz': 0,
                     'rxy': 0,
                     'rxz': 0,
                     }
                  },
                 {'n_ex_per_model': 30,
                  'name': 'rotation', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': 1,
                     'ty': 0,
                     'tz': 0,
                     'ryz': uniform(-180., 180.),
                     'rxy': uniform(-180., 180.),
                     'rxz': uniform(-180., 180.),
                     }
                  },
                 {'n_ex_per_model': 100,
                  'name': 'var1', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': loguniform(np.log(2./3), np.log(2.)),
                     'ty': uniform(-1.0, 1.0),
                     'tz': uniform(-1.0, 1.0),
                     'ryz': uniform(-180., 180.),
                     'rxy': uniform(-180., 180.),
                     'rxz': uniform(-180., 180.),
                     }
                  }]
    specific_name = 'GenerativeDataset1'


class GenerativeDataset2(GenerativeDatasetBase):    
    models = model_info.MODEL_SUBSET_1
    bad_backgrounds = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg',
                       'INTERIOR_11ST.jpg', 'INTERIOR_10ST.jpg',
                       'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
                       'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg',
                       'INTERIOR_05ST.jpg']
    good_backgrounds = [_b for _b in model_info.BACKGROUNDS
                                                  if _b not in bad_backgrounds]
    templates = [{'n_ex_per_model': 10,
                  'name': 'var0',  
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': 1,
                     'ty': 0,
                     'tz': 0,
                     'ryz': 0,
                     'rxy': 0,
                     'rxz': 0,
                     }
                 },
                 {'n_ex_per_model': 150,
                  'name': 'var1', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': loguniform(np.log(2./3), np.log(2.)),
                     'ty': uniform(-1.0, 1.0),
                     'tz': uniform(-1.0, 1.0),
                     'ryz': uniform(-180., 180.),
                     'rxy': uniform(-180., 180.),
                     'rxz': uniform(-180., 180.),
                     }
                  }]
    specific_name = 'GenerativeDataset2'

    

class GenerativeDatasetTest(GenerativeDataset1):    
    models = model_info.MODEL_SUBSET_1
    bad_backgrounds = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg',
                       'INTERIOR_11ST.jpg', 'INTERIOR_10ST.jpg',
                       'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
                       'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg',
                       'INTERIOR_05ST.jpg']
    good_backgrounds = [_b for _b in model_info.BACKGROUNDS
                                                  if _b not in bad_backgrounds]
    templates = [{'n_ex_per_model': 10,
                  'name': 'var1', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': loguniform(np.log(2./3), np.log(2.)),
                     'ty': uniform(-1.0, 1.0),
                     'tz': uniform(-1.0, 1.0),
                     'ryz': uniform(-180., 180.),
                     'rxy': uniform(-180., 180.),
                     'rxz': uniform(-180., 180.),
                     }
                  }]  
    specific_name = 'GenerativeDatasetTest'
    

class ImgRendererResizer(object):
    def __init__(self, model_root, bg_root, preproc, lbase, output):
        self._shape = tuple(preproc['size'])
        self._ndim = len(self._shape) 
        self._dtype = preproc['dtype']
        self.mode = preproc['mode']
        self.normalize = preproc['normalize']
        self.lbase = lbase
        self.output = output
        self.model_root = model_root
        self.bg_root = bg_root
    
    def rval_getattr(self, attr, objs):
        if attr == 'shape' and self._shape is not None:
            return self._shape
        if attr == 'ndim' and self._ndim is not None:
            return self._ndim
        if attr == 'dtype':
            return self._dtype
        raise AttributeError(attr)
        
    def __call__(self, m):
        modelpath = os.path.join(self.model_root, 
                                 m['obj'], m['obj'] + '.bam')
        bgpath = os.path.join(self.bg_root, m['bgname'])
        scale = [m['s']]
        pos = [m['ty'], m['tz']]
        hpr = [m['ryz'], m['rxz'], m['rxy']]
        bgscale = [m['bgscale']]
        bghp = [m['bgphi'], m['bgpsi']]
        args = (modelpath, bgpath, scale, pos, hpr, bgscale, bghp)
        objnode, bgnode = gr.construct_scene(self.lbase, *args)
        self.lbase.render_frame()
        objnode.removeNode()
        bgnode.removeNode()
        tmpfilename = get_tmpfilename() + '.png'
        self.lbase.screenshot(self.output, pth=tmpfilename)
        im = Image.open(tmpfilename)
        if im.mode != self.mode:
            im = im.convert(self.mode)
        rval = np.asarray(im, self._dtype)
        if self.normalize:
            rval -= rval.mean()
            rval /= max(rval.std(), 1e-3)
        else:
            rval /= 255.0
        assert rval.shape[:2] == self._shape[:2]
        os.remove(tmpfilename)
        return rval
        

class TrainingDataset(DatasetBase):

    FILES = [('genthor_training_data_20120416.zip',
              'cc5cbb5fd25cb469783e2494d7efdf1d189035a5')]

    name = 'GenthorTrainingDataset'
    
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
                                             'obj',
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

    def get_images(self, preproc):
        self.fetch()
        size = tuple(preproc['size'])
        normalize = preproc['global_normalize']
        mode = preproc['mode']
        dtype = preproc['dtype']
        return larray.lmap(ImgLoaderResizer(inshape=(256, 256),
                                            shape=size,
                                            dtype=dtype,
                                            normalize=normalize,
                                            mode=mode),
                                self.filenames)


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
        self.inshape = tuple(inshape)
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
    
    
def test_training_dataset():
    dataset = TrainingDataset()
    meta = dataset.meta
    assert len(meta) == 11000
    agg = meta[['obj', 'category']].aggregate(['category'],
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

    agg2 = meta[['obj', 'category']].aggregate(['category'], 
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


##XXX TO DO:  TEST simultaneous reads more thoroughly
def test_generative_dataset():
    dataset = GenerativeDatasetTest()
    meta = dataset.meta
    ids = cPickle.load(open('dataset_ids.pkl'))
    assert (meta['id'] == ids).all()
    S, v = dataset.get_subset_splits(20, [10], 5, 
                            lambda x : x['category'], None, None, None, 0)
    assert len(S) == 5
    for s in S:
        assert sorted(s.keys()) == ['test', 'train']
        assert len(s['train']) == 220
        assert len(s['test']) == 110 
        assert set(s['train']).intersection(s['test']) == set([])
    
    imgs = dataset.get_images({'size':(256, 256),
               'mode': 'L', 'normalize': False, 'dtype':'float32'})
    X = np.asarray(imgs[[0, 50]])
    Y = cPickle.load(open('generative_dataset_test_images_0_50.pkl'))
    assert (X == Y).all()
