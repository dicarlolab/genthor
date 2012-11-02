import os
import itertools
import re
import hashlib
import cPickle

import lockfile
import numpy as np
import Image
import tabular as tb
from yamutils.fast import reorder_to, isin
from yamutils.basic import dict_inverse
import boto

import pyll
choice = pyll.scope.choice
uniform = pyll.scope.uniform
loguniform = pyll.scope.loguniform
import pyll.stochastic as stochastic

import skdata.larray as larray
from skdata.data_home import get_data_home
from skdata.utils.download_and_extract import extract, download

import genthor as gt
import genthor.renderer.renderer as gr
import genthor.model_info as model_info
import genthor.jxx_model_info as jxx_model_info
from genthor.renderer.imager import Imager
import genthor.tools as tools
import genthor.modeltools.tools as mtools

import pdb


class DatasetBase(object):
    RESOURCE_PATH = gt.RESOURCE_PATH
    OBJ_PATH = gt.OBJ_PATH
    BACKGROUND_PATH = gt.BACKGROUND_PATH
    CACHE_PATH = gt.CACHE_PATH
    
    def resource_home(self, *suffix_paths):
        return os.path.join(self.RESOURCE_PATH, *suffix_paths)

    def obj_home(self, *suffix_paths):
        return os.path.join(self.OBJ_PATH, *suffix_paths)

    def cache_home(self, *suffix_paths):
        return os.path.join(self.CACHE_PATH, *suffix_paths)

    def fetch(self):
        """Download and extract the dataset."""
        resource_home = self.resource_home()
        if not os.path.exists(resource_home):
            os.makedirs(resource_home)
        cachedir = self.cache_home()
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)      

        lock = lockfile.FileLock(resource_home)
        with lock:
            tools.download_s3_directory(gt.s3_resource_bucket,
                                        resource_home)

    def get_model(self, name):
        dirn = self.obj_home(name)
        path = os.path.join(dirn, name + '.tgz')
        old_model_bucket = gt.s3_old_model_bucket
        conn = boto.connect_s3()
        k = conn.get_bucket(old_model_bucket).get_key(name + '.tar.gz')
        if k is not None:
            os.mkdir(dirn)
            print('downloading %s' % k.name)
            k.get_contents_to_filename(path)
        else:
            raise ValueError("didn't find key %s" % name)  
        #tools.upload_s3_directory(gt.s3_resource_bucket, dir)
        
    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self.fetch()
            self._meta = self._get_meta()
        return self._meta

    def get_subset_splits(self, *args, **kwargs):
        return get_subset_splits(self.meta, *args, **kwargs) 


class GenerativeBase(DatasetBase):
    #must subclass this and define a ._get_meta method the constructs the 
    #metadata tabarray.   

    check_penetration = False

    def __init__(self, data=None):
        self.data = data
        self.specific_name = self.__class__.__name__ + '_' + get_image_id(data)
        model_root = self.OBJ_PATH
        bg_root = self.BACKGROUND_PATH
        self.imager = Imager(model_root, bg_root, 
                             check_penetration=self.check_penetration)
    
    def get_image(self, preproc, config):
        if not isinstance(config['obj'], list):
            dname = [self.obj_home(config['obj'])]
            cobj = [config['obj']]
        else:
            dname = [self.obj_home(co) for co in config['obj']]
            cobj = config['obj']
        for d, co in zip(dname, cobj):
            if not os.path.exists(d):
                self.get_model(co)
        irr = self.imager.get_map(preproc, 'texture')
        return irr(config)

    def get_images(self, preproc):
        name = self.specific_name + '_' + get_image_id(preproc)
        cachedir = self.cache_home()
        meta = self.meta
        window_type = 'texture'
        size = preproc['size']
        irr = self.imager.get_map(preproc, window_type)
        image_map = larray.lmap(irr, meta)
        return larray.cache_memmap(image_map, name=name, basedir=cachedir)


class GenerativeDatasetBase(GenerativeBase):
    """A class that generates randomly sampled metadata for single objects 
    from a set of templates.   Datasets are implemented as subclasses of this 
    class which define the "templates" attribute 
    as class attributes
    """
    model_categories = dict_inverse(model_info.MODEL_CATEGORIES)
    
    def _get_meta(self):
        #generate params 
        models = self.models
        templates = self.templates

        latents = []
        rng = np.random.RandomState(seed=0)
        model_categories = self.model_categories
        for tdict in templates:
            template = tdict['template']
            tname = tdict['name']
            if tdict.has_key('n_ex_dict'):
                n_ex_dict = tdict['n_ex_dict']
            else:
                n_ex_dict = dict([(m, tdict['n_ex_per_model']) for m in models])
            for model in models:
                print('Generating meta for %s' % model)
                for _ind in range(n_ex_dict[model]):
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
        
        
class GenerativeMultiDatasetTest(GenerativeDatasetBase):
    """multi-object rendering dataset
    """
    check_penetration = True

    def _get_meta(self):
        #generate params 
        
        bgname = [model_info.BACKGROUNDS[0],
                  model_info.BACKGROUNDS[0]]
        bgphi = [0, 0]
        bgpsi = [0, 0]
        bgscale = [1, 1]
        ty = [[0, .2], [0, 0]]
        tz = [[-0.2, 0.2], [0, 0]]
        s = [[1, 1], [1, 1]]
        ryz = [[0, 0], [0, 0]]
        rxz = [[0, 0], [0, 90]]
        rxy = [[0, 0], [0, 0]]
        obj = [['MB26897', 'MB28049'], ['MB26897', 'MB28049']]
        category = [['cars', 'tables'], ['cars', 'tables']]
        latents = zip(*[bgname, bgphi, bgpsi, bgscale, obj, category,
                   ryz, rxz, rxy, ty, tz, s, ['t0', 't0'], ['testing', 'testing']])
        

        meta = tb.tabarray(records=latents, names = ['bgname',
                                                     'bgphi',
                                                     'bgpsi',
                                                     'bgscale',
                                                     'obj',
                                                     'category',
                                                     'ryz',
                                                     'rxz',
                                                     'rxy',
                                                     'ty',
                                                     'tz',
                                                     's',
                                                     'tname',
                                                     'id'], 
                           formats=['|S20', 'float', 'float', 'float'] + \
                                  ['|O8']*8 +  ['|S10', '|S10'])
        return meta


class GenerativeEmptyDatasetTest(GenerativeDatasetBase):
    """rendering empty frame with just background
    """

    def _get_meta(self):
        #generate params 
        
        bgname = [model_info.BACKGROUNDS[0]]
        bgphi = [0]
        bgpsi = [0]
        bgscale = [1]
        ty = [[]]
        tz = [[]]
        s = [[]]
        ryz = [[]]
        rxz = [[]]
        rxy = [[]]
        obj = [[]]
        category = [[]]
        latents = zip(*[bgname, bgphi, bgpsi, bgscale, obj, category,
                   ryz, rxz, rxy, ty, tz, s, ['t0'], ['testing']])
        

        meta = tb.tabarray(records=latents, names = ['bgname',
                                                     'bgphi',
                                                     'bgpsi',
                                                     'bgscale',
                                                     'obj',
                                                     'category',
                                                     'ryz',
                                                     'rxz',
                                                     'rxy',
                                                     'ty',
                                                     'tz',
                                                     's',
                                                     'tname',
                                                     'id'], 
                           formats=['|S20', 'float', 'float', 'float'] + \
                                  ['|O8']*8 +  ['|S10', '|S10'])
        return meta
    

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


class GenerativeDataset3(GenerativeDatasetBase):    
    models = model_info.MODEL_SUBSET_3
    bad_backgrounds = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg',
                       'INTERIOR_11ST.jpg', 'INTERIOR_10ST.jpg',
                       'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
                       'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg',
                       'INTERIOR_05ST.jpg']
    good_backgrounds = [_b for _b in model_info.BACKGROUNDS
                                                  if _b not in bad_backgrounds]
    templates = [
                 {'n_ex_per_model': 200,
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


class GenerativeDataset3(GenerativeDatasetBase):    
    models = model_info.MODEL_SUBSET_3
    bad_backgrounds = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg',
                       'INTERIOR_11ST.jpg', 'INTERIOR_10ST.jpg',
                       'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
                       'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg',
                       'INTERIOR_05ST.jpg']
    good_backgrounds = [_b for _b in model_info.BACKGROUNDS
                                                  if _b not in bad_backgrounds]
    templates = [
                 {'n_ex_per_model': 200,
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


class GenerativeDatasetTestAll(GenerativeDatasetBase):    
    models = list(itertools.chain(*model_info.MODEL_CATEGORIES.values()))
    bad_backgrounds = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg',
                       'INTERIOR_11ST.jpg', 'INTERIOR_10ST.jpg',
                       'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
                       'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg',
                       'INTERIOR_05ST.jpg']
    good_backgrounds = [_b for _b in model_info.BACKGROUNDS
                                                  if _b not in bad_backgrounds]
    templates = [
                 {'n_ex_per_model': 1,
                  'name': 'var1', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': 1.75,
                     'ty': 0,
                     'tz': 0,
                     'ryz': 0,
                     'rxy': 0,
                     'rxz': 0,
                     }
                  },
                  {'n_ex_per_model': 1,
                  'name': 'var1', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': 1.75,
                     'ty': 0,
                     'tz': 0,
                     'ryz': 45,
                     'rxy': 45,
                     'rxz': 45,
                     }}
                   ]
    

class GenerativeDataset4(GenerativeDatasetBase):    
    models = model_info.MODEL_SUBSET_5
    bad_backgrounds = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg',
                       'INTERIOR_11ST.jpg', 'INTERIOR_10ST.jpg',
                       'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
                       'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg',
                       'INTERIOR_05ST.jpg']
    good_backgrounds = [_b for _b in model_info.BACKGROUNDS
                                                  if _b not in bad_backgrounds]
    templates = [
                 {'n_ex_per_model': 250,
                  'name': 'var1', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': uniform(2./3, 3),
                     'ty': uniform(-1.0, 1.0),
                     'tz': uniform(-1.0, 1.0),
                     'ryz': uniform(-180., 180.),
                     'rxy': uniform(-180., 180.),
                     'rxz': uniform(-180., 180.),
                     }
                  }]
    
    def __init__(self, data=None):
        GenerativeDatasetBase.__init__(self, data)
        if self.data and self.data.get('bias_file') is not None:
            froot = os.environ.get('FILEROOT','')
            bias = cPickle.load(open(os.path.join(froot, self.data['bias_file'])))
        elif self.data and self.data.get('bias') is not None:
            bias = self.data['bias']
        else:
            bias = None
        if self.data and self.data.get('n_ex_per_model'):
            self.templates[0]['n_ex_per_model'] = self.data['n_ex_per_model']
        if bias is not None:
            models = self.models
            n_ex = self.templates[0]['n_ex_per_model']
            total = len(models) * n_ex
            self.templates[0]['n_ex_dict'] = dict(zip(models,
                               [int(round(total * bias[m])) for m in models]))


class GenerativeDataset5(GenerativeDataset4):   
    models = model_info.MODEL_SUBSET_5
    bad_backgrounds = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg',
                       'INTERIOR_11ST.jpg', 'INTERIOR_10ST.jpg',
                       'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
                       'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg',
                       'INTERIOR_05ST.jpg']
    good_backgrounds = [_b for _b in model_info.BACKGROUNDS
                                                  if _b not in bad_backgrounds]

    templates = [
                 {'n_ex_per_model': 250,
                  'name': 'var1', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': uniform(2./3, 3),
                     'ty': uniform(-0.5, 0.5),
                     'tz': uniform(-0.5, 0.5),
                     'ryz': uniform(-180., 180.),
                     'rxy': uniform(-180., 180.),
                     'rxz': uniform(-180., 180.),
                     }
                  }]   


class GenerativeDatasetLoTrans(GenerativeDataset4):   
    models = model_info.MODEL_SUBSET_5
    bad_backgrounds = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg',
                       'INTERIOR_11ST.jpg', 'INTERIOR_10ST.jpg',
                       'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
                       'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg',
                       'INTERIOR_05ST.jpg']
    good_backgrounds = [_b for _b in model_info.BACKGROUNDS
                                                  if _b not in bad_backgrounds]

    templates = [
                 {'n_ex_per_model': 250,
                  'name': 'var1', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': uniform(2./3, 3),
                     'ty': uniform(-0.05, 0.05),
                     'tz': uniform(-0.05, 0.05),
                     'ryz': uniform(-180., 180.),
                     'rxy': uniform(-180., 180.),
                     'rxz': uniform(-180., 180.),
                     }
                  }]   


class GenerativeDatasetLoTransSmall(GenerativeDataset4):   
    models = model_info.MODEL_SUBSET_5[:2]
    bad_backgrounds = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg',
                       'INTERIOR_11ST.jpg', 'INTERIOR_10ST.jpg',
                       'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
                       'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg',
                       'INTERIOR_05ST.jpg']
    good_backgrounds = [_b for _b in model_info.BACKGROUNDS
                                                  if _b not in bad_backgrounds]

    templates = [
                 {'n_ex_per_model': 5,
                  'name': 'var1', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': uniform(2./3, 3),
                     'ty': uniform(-0.05, 0.05),
                     'tz': uniform(-0.05, 0.05),
                     'ryz': uniform(-180., 180.),
                     'rxy': uniform(-180., 180.),
                     'rxz': uniform(-180., 180.),
                     }
                  }]   



class GenerativeDatasetHiZTrans(GenerativeDataset4):   
    models = model_info.MODEL_SUBSET_5
    bad_backgrounds = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg',
                       'INTERIOR_11ST.jpg', 'INTERIOR_10ST.jpg',
                       'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
                       'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg',
                       'INTERIOR_05ST.jpg']
    good_backgrounds = [_b for _b in model_info.BACKGROUNDS
                                                  if _b not in bad_backgrounds]

    templates = [
                 {'n_ex_per_model': 250,
                  'name': 'var1', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': uniform(2./3, 3),
                     'ty': uniform(-0.05, 0.05),
                     'tz': uniform(-5.0, 5.0),
                     'ryz': uniform(-180., 180.),
                     'rxy': uniform(-180., 180.),
                     'rxz': uniform(-180., 180.),
                     }
                  }]   


class GenerativeDatasetBoatsVsAll(GenerativeDatasetBase):    
    models = model_info.MODEL_SUBSET_5
    bad_backgrounds = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg',
                       'INTERIOR_11ST.jpg', 'INTERIOR_10ST.jpg',
                       'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
                       'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg',
                       'INTERIOR_05ST.jpg']
    good_backgrounds = [_b for _b in model_info.BACKGROUNDS
                                                  if _b not in bad_backgrounds]
    templates = [
                 {'n_ex_dict': dict([(m, 1125 if m in model_info.MODEL_CATEGORIES['boats'] else 140) for m in models]),
                  'name': 'var1', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': uniform(2./3, 3),
                     'ty': uniform(-1.0, 1.0),
                     'tz': uniform(-1.0, 1.0),
                     'ryz': uniform(-180., 180.),
                     'rxy': uniform(-180., 180.),
                     'rxz': uniform(-180., 180.),
                     }
                  }]


class GenerativeDatasetTwoBadBoats(GenerativeDatasetBase):    
    models = model_info.MODEL_SUBSET_5[:]
    models.remove('MB27840')
    models.remove('MB28586')
    bad_backgrounds = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg',
                       'INTERIOR_11ST.jpg', 'INTERIOR_10ST.jpg',
                       'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
                       'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg',
                       'INTERIOR_05ST.jpg']
    good_backgrounds = [_b for _b in model_info.BACKGROUNDS
                                                  if _b not in bad_backgrounds]
    templates = [
                 {'n_ex_dict': dict([('MB28646', 1500), ('MB29346', 1500)] + \
                                    [(m , 94) for m in models if m not in model_info.MODEL_CATEGORIES['boats']]),
                  'name': 'var1', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': uniform(2./3, 3),
                     'ty': uniform(-1.0, 1.0),
                     'tz': uniform(-1.0, 1.0),
                     'ryz': uniform(-180., 180.),
                     'rxy': uniform(-180., 180.),
                     'rxz': uniform(-180., 180.),
                     }
                  }]


class GenerativeDatasetPlanesVsAll(GenerativeDatasetBase):    
    models = model_info.MODEL_SUBSET_5
    bad_backgrounds = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg',
                       'INTERIOR_11ST.jpg', 'INTERIOR_10ST.jpg',
                       'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
                       'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg',
                       'INTERIOR_05ST.jpg']
    good_backgrounds = [_b for _b in model_info.BACKGROUNDS
                                                  if _b not in bad_backgrounds]
    templates = [
                 {'n_ex_dict': dict([(m, 750 if m in model_info.MODEL_CATEGORIES['planes'] else 93) for m in models]),
                  'name': 'var1', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': uniform(2./3, 3),
                     'ty': uniform(-1.0, 1.0),
                     'tz': uniform(-1.0, 1.0),
                     'ryz': uniform(-180., 180.),
                     'rxy': uniform(-180., 180.),
                     'rxz': uniform(-180., 180.),
                     }
                  }]


class GenerativeDatasetTablesVsAll(GenerativeDatasetBase):    
    models = model_info.MODEL_SUBSET_5
    bad_backgrounds = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg',
                       'INTERIOR_11ST.jpg', 'INTERIOR_10ST.jpg',
                       'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
                       'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg',
                       'INTERIOR_05ST.jpg']
    good_backgrounds = [_b for _b in model_info.BACKGROUNDS
                                                  if _b not in bad_backgrounds]
    templates = [
                 {'n_ex_dict': dict([(m, 750 if m in model_info.MODEL_CATEGORIES['tables'] else 93) for m in models]),
                  'name': 'var1', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': uniform(2./3, 3),
                     'ty': uniform(-1.0, 1.0),
                     'tz': uniform(-1.0, 1.0),
                     'ryz': uniform(-180., 180.),
                     'rxy': uniform(-180., 180.),
                     'rxz': uniform(-180., 180.),
                     }
                  }]


MODEL_CATEGORIES = model_info.MODEL_CATEGORIES

class GenerativeDatasetBoatsVsReptiles(GenerativeDatasetBase):    
    models = [_x for _x in model_info.MODEL_SUBSET_5 
                     if _x in MODEL_CATEGORIES['boats'] + MODEL_CATEGORIES['reptiles']]
    bad_backgrounds = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg',
                       'INTERIOR_11ST.jpg', 'INTERIOR_10ST.jpg',
                       'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
                       'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg',
                       'INTERIOR_05ST.jpg']
    good_backgrounds = [_b for _b in model_info.BACKGROUNDS
                                                  if _b not in bad_backgrounds]
    templates = [
                 {'n_ex_per_model': 1000,
                  'name': 'var1', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': uniform(2./3, 3),
                     'ty': uniform(-1.0, 1.0),
                     'tz': uniform(-1.0, 1.0),
                     'ryz': uniform(-180., 180.),
                     'rxy': uniform(-180., 180.),
                     'rxz': uniform(-180., 180.),
                     }
                  }]


class GenerativeDatasetLowres(GenerativeDatasetBase):    
    """optimized for low res:, e.g. bigger objects"""
    models = model_info.MODEL_SUBSET_3
    bad_backgrounds = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg',
                       'INTERIOR_11ST.jpg', 'INTERIOR_10ST.jpg',
                       'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
                       'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg',
                       'INTERIOR_05ST.jpg']
    good_backgrounds = [_b for _b in model_info.BACKGROUNDS
                                                  if _b not in bad_backgrounds]
    templates = [
                 {'n_ex_per_model': 250,
                  'name': 'var1', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': loguniform(np.log(1.5), np.log(5.)),
                     'ty': uniform(-0.2, 0.2),
                     'tz': uniform(-0.2, 0.2),
                     'ryz': uniform(-45., 45.),
                     'rxy': uniform(-45., 45.),
                     'rxz': uniform(-45., 45.),
                     }
                  }]

    
    
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
    
    
#####GP generative
class GPGenerativeDatasetBase(GenerativeDatasetBase):

    def _get_meta(self, seed=0):
        #generate params
        rng = np.random.RandomState(seed=seed)
        
        models = self.models
        template = self.template

        model_categories = dict_inverse(model_info.MODEL_CATEGORIES)
        
        import sklearn.gaussian_process as gaussian_process 
        
        gps = self.gps = [gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4,
                     thetaU=1e-1, corr='linear')  for _i in range(len(models))]
        
        data = self.data
        X, y = data['bias_data']
        M = data['num_to_sample']
        N = data['num_images']
        
        [self.gps[i].fit(X[i], y[i]) for i in range(len(models))]
        
        mx = X.max(1)
        mn = X.min(1)
        Ts = [rng.uniform(size=(M, 6)) * (mx[i] - mn[i]) + mn[i] for i in range(len(models))]
        Tps = [gps[i].predict(Ts[i]) for i in range(len(models))]
        Tps = [np.minimum(t, 0) for t in Tps]
        Tps = [(t / t.sum()) * y[i].sum() for i, t in enumerate(Tps)]
        
        W = tb.tab_rowstack([tb.tabarray(records=[(tt, i, j) for (j, tt) in enumerate(t)],
                   names=['w', 'o', 'j']) for i, t in enumerate(Tps)])
    
        L = sample_without_replacement(W['w'], N, rng)
        
        latents = []
        for w in W[L]:
            obj = models[w['o']]
            cat = model_categories[obj][0]
            l = Ts[w['o']][w['j']]
            l1 = stochastic.sample(template, rng)
            rec = (l1['bgname'],
                   float(l1['bgphi']),
                   float(l1['bgpsi']),
                   float(l1['bgscale']),
                   cat,
                   obj) + tuple(l)
            idval = get_image_id(rec)
            rec = rec + (idval,)
            latents.append(rec)

        return tb.tabarray(records=latents, names = ['bgname',
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
                                                     'id'])
        
def sample_without_replacement(w, N, rng):
    w = w.copy()
    assert (w >= 0).all()
    assert np.abs(w.sum() - 1) < 1e-4, w.sum()
    assert w.ndim == 1
    assert len((w > 0).nonzero()[0]) >= N, (len((w >0).nonzero()[0]), N)
    samples = []
    for ind in xrange(N):
        r = rng.uniform()
        j = w.cumsum().searchsorted(r)
        samples.append(j)
        w[j] = 0
        w = w / w.sum()
    return samples


class GPGenerativeDatasetTest(GPGenerativeDatasetBase):
    models = GenerativeDataset4.models[:]
    good_backgrounds = GenerativeDataset4.good_backgrounds[:]
    template = {'bgname': choice(good_backgrounds),
                'bgscale': 1.,
                'bgpsi': 0,
                'bgphi': uniform(-180.0, 180.)}


class ResampleGenerativeDataset(GenerativeDatasetBase):
    def _get_meta(self, seed=0):
        #generate params
        rng = np.random.RandomState(seed=seed)                
        data = self.data
        bias_meta, bias_weights = data['bias_data']
        ranges = data['ranges']
        N = data['num_images']

        J = sample_with_replacement(bias_weights, N, rng)

        latents = []        
        for j in J:
            l = get_nearby_sample(bias_meta[j], ranges, rng)     
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
                                                     'id'])
        return meta


def sample_with_replacement(w, N, rng):
    assert (w >= 0).all()
    assert np.abs(w.sum() - 1) < 1e-4, w.sum()
    assert w.ndim == 1
    return w.cumsum().searchsorted(rng.uniform(size=(N,)))
    

def get_nearby_sample(s, ranges, rng):
    news = {}
    news['bgname'] = s['bgname']
    news['category'] = s['category']
    news['obj'] = s['obj']
    post = {'bgphi': lambda x: mod(x, 360, 180),
            'bgpsi': lambda x: mod(x, 360, 180),
            'rxy': lambda x: mod(x, 360, 180),
            'ryz': lambda x: mod(x, 360, 180),
            'rxz': lambda x: mod(x, 360, 180)}
    for k in ['bgphi', 'bgpsi', 'bgscale', 'rxy', 'rxz', 'ryz', 'ty', 'tz', 's']:
        delta = rng.uniform(high=ranges[k][1], low=ranges[k][0])
        news[k] = post.get(k, lambda x: x)(s[k] + delta)
    return news    
                
    
def mod (x, y, a):
    return (x + a) % y - a


class ResampleGenerativeDataset4a(ResampleGenerativeDataset):    
    def _get_meta(self):
        dset = GenerativeDataset4()
        dset.templates[0]['n_ex_per_model'] = 125
        meta1 = dset.meta
        dset = GenerativeDataset4()
        dset.templates[0]['n_ex_per_model'] = 250
        meta = dset.meta
        froot = os.environ.get('FILEROOT','')
        bias = cPickle.load(open(os.path.join(froot, self.data['bias_file'])))
        self.data['bias_data'] = (meta, bias)
        self.data['num_images'] = len(meta)/2
        meta2 = ResampleGenerativeDataset._get_meta(self)
        return tb.tab_rowstack([meta1, meta2])


class ResampleGenerativeDataset4plus(ResampleGenerativeDataset):    
    def _get_meta(self):
        dset = GenerativeDataset4()
        dset.templates[0]['n_ex_per_model'] = 125
        meta1 = dset.meta
        froot = os.environ.get('FILEROOT','')
        self.data['bias_data'] = cPickle.load(open(os.path.join(froot,
                       self.data['bias_file'])))
        self.data['num_images'] = len(meta1)
        meta2 = ResampleGenerativeDataset._get_meta(self)
        return tb.tab_rowstack([meta1, meta2])


class GenerativeDatasetLowres(GenerativeDatasetBase):    
    """optimized for low res:, e.g. bigger objects"""
    models = model_info.MODEL_SUBSET_3
    bad_backgrounds = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg',
                       'INTERIOR_11ST.jpg', 'INTERIOR_10ST.jpg',
                       'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
                       'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg',
                       'INTERIOR_05ST.jpg']
    good_backgrounds = [_b for _b in model_info.BACKGROUNDS
                                                  if _b not in bad_backgrounds]
    templates = [
                 {'n_ex_per_model': 250,
                  'name': 'var1', 
                  'template': {'bgname': choice(good_backgrounds),
                     'bgscale': 1.,
                     'bgpsi': 0,
                     'bgphi': uniform(-180.0, 180.),
                     's': loguniform(np.log(1.5), np.log(5.)),
                     'ty': uniform(-0.2, 0.2),
                     'tz': uniform(-0.2, 0.2),
                     'ryz': uniform(-45., 45.),
                     'rxy': uniform(-45., 45.),
                     'rxz': uniform(-45., 45.),
                     }
                  }]
