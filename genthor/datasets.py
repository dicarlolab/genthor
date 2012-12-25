import os
import itertools
import re
import hashlib
import cPickle
import copy
import pymongo
import json
import urllib
import functools

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
from skdata.utils.download_and_extract import extract, download, download_boto

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
    HUMAN_PATH = gt.HUMAN_PATH
    HUMAN_DATA = []
    
    def resource_home(self, *suffix_paths):
        return os.path.join(self.RESOURCE_PATH, *suffix_paths)

    def obj_home(self, *suffix_paths):
        return os.path.join(self.OBJ_PATH, *suffix_paths)

    def cache_home(self, *suffix_paths):
        return os.path.join(self.CACHE_PATH, *suffix_paths)
        
    def human_home(self, *suffix_paths):
        return os.path.join(self.HUMAN_PATH, *suffix_paths)

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
        if not os.path.exists(self.human_home()):
            os.makedirs(self.human_home())            
        for x, n, task, sha1 in self.HUMAN_DATA:
            filename = self.human_home(x.split('/')[-1])
            print(filename)
            if not os.path.exists(filename):
                url = 'http://dicarlocox-datasets.s3.amazonaws.com/' + x
                print ('downloading %s' % url)
                download_boto(url, (None, ), filename, sha1=sha1)

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

    def get_models(self):
        objs = self.objects
        for o in objs:
            pth = self.obj_home(o)
            if not os.path.exists(pth):
                self.get_model(o)

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self.fetch()
            self._meta = self._get_meta()
        return self._meta

    @property
    def objects(self):
        objs = self.meta['obj'].tolist()
        objs = set(objs)
        uobjs = []
        for o in objs:
            if isstring(o):
                uobjs.append(o)
            else:
                uobjs.extend(o)
        return uobjs
                
    def get_subset_splits(self, *args, **kwargs):
        return get_subset_splits(self.meta, *args, **kwargs) 
        
    def human_data(self, taskq):
 
        if not hasattr(self, '_human_data'):
            hd = copy.deepcopy(self.HUMAN_DATA)
            for _i, hde in enumerate(hd):
                hd[_i] = (self.human_home(hde[0].split('/')[-1]),) + hde[1:]        
            self.fetch()
            self._human_data = parse_human_data(hd)
        return get_matching_tasks(self.meta, self._human_data, taskq)

    def human_confusion_mat_by_task(self, task):
        hds = self.human_data(task)
        assert len(hds) == 1
        hdn, hd = hds[0]
        hdd = hd['data']
        hdt = hd['task']

        meta = self.meta
        fns = np.array([fn.split('/')[-1] for fn in meta['id']])
        st = fns.argsort()
        meta = meta[st]
        fns = fns[st]
        labelfunc = get_labelfunc_from_config(task['labelfunc'])
        labels_all, _ = labelfunc(meta)

        train_q = get_lambda_from_query_config(hdt['train_q'])
        inds_train = np.array(map(train_q, meta)).astype(np.bool)
        uniques = np.unique(labels_all[inds_train])

        cms = []
        for rec in hdd:
            ss = rec['StimShown']
            response = rec['Response']
            fns0 = np.array([fn.split('/')[-1].split('.')[0] for fn in ss])
            inds = np.searchsorted(fns, fns0)
            #actual = labels_all[inds]
            actual = labels_all[inds][:len(response)]
            #print(len(actual), len(response), hdn)
            cms.append([[((actual == v1) & (response == v2)).sum() for v1 in uniques] for v2 in uniques])

        #cms.shape = (actual, grnd truth, subjects)
        cms = np.array(cms).T

        return cms, uniques


def get_matching_tasks(meta, taskdict, taskq):
    matchlist = []
    for tn, tdata in taskdict.items():
        tspec = tdata['task']
        if taskq.has_key('split_by') and (tspec['split_by'] != taskq['split_by']):
            continue
        if taskq.has_key('labelfunc') and (tspec['labelfunc'] != taskq['labelfunc']):
            continue
        if taskq.has_key('train_q') and (tspec['train_q'] != taskq['train_q']):
            continue
        if taskq.has_key('test_q') and (tspec['test_q'] != taskq['test_q']):
            continue
        matchlist.append((tn, tdata))
    return matchlist
    

# copied from new_new_bandits.py
def kfunc(k, x):
    U = np.sort(np.unique(x[k]))
    return x[k], U   # U is added for consistency with kfuncb


# copied from new_new_bandits.py
def kfuncb(k, x, U=None):
    if U is None:
        U = np.sort(np.unique(x[k]))
    else:
        assert set(U) == set(np.sort(np.unique(x[k])))
    return np.array([x[k] == u for u in U]).T, U

LABEL_REGISTRY = dict([(k, functools.partial(kfunc, k))
                for k in ['obj', 'category', 'ty','tz','s', 'ryz', 'rxy', 'rxz']]
                  + [(k + '_binary', functools.partial(kfuncb, k))
                for k in ['obj', 'category']]
                  )


def get_labelfunc_from_config(q):
    """turns a dictionary into a function that returns a label
    for each record of a metadata table
    """
    return LABEL_REGISTRY[q]


def get_lambda_from_query_config(q):
    """turns a dictionary specificying a mongo query (basically)
    into a lambda for subsetting a data table

    TODO: implement OR or not, etc.
    See: https://github.com/yamins81/devthor/commit/367d9e0714d5d89dc08c4e37d653d716c87b64be#commitcomment-1657582
    """
    return lambda x:  all([x[k] in v for k, v in q.items()])   # per Dan's suggestion..


class GenerativeBase(DatasetBase):
    #must subclass this and define a ._get_meta method the constructs the 
    #metadata tabarray.   

    check_penetration = False

    def __init__(self, data=None, **kwargs):
        if data is None:
            data = {}
        data.update(kwargs)
        self.data = data
        self.specific_name = self.__class__.__name__ + '_' + get_image_id(data)
        model_root = self.OBJ_PATH
        bg_root = self.BACKGROUND_PATH
        self.imager = Imager(model_root, bg_root, 
                             check_penetration=self.check_penetration)
        self.noise = kwargs.get('noise')
            
        self.irrs = {}
        if 'dbname' in kwargs:
            self.dbname = kwargs['dbname']
            self.colname = kwargs['colname']
            self.hostname = kwargs['hostname']
            self.port = kwargs['port']
            self.use_canonical = kwargs['use_canonical']
            self.canonical_user = kwargs['canonical_user']
            self.conn = pymongo.Connection(port=self.port, host=self.hostname) #Open connection
            self.db = self.conn[self.dbname]
            self.col = self.db[self.colname]
            self.col.ensure_index([('obj', pymongo.ASCENDING),
                                     ('user', pymongo.ASCENDING),
                                     ('version', pymongo.DESCENDING)],
                                  unique=True)
        else:
            self.use_canonical = False
        self.internal_canonical = kwargs.get('internal_canonical', False)
    
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
        phash = json.dumps(preproc)
        if phash not in self.irrs:
            self.irrs[phash] = self.imager.get_map(preproc, 'texture')
        irr = self.irrs[phash]
        use_canonical = config.get('use_canonical', self.use_canonical)
        if use_canonical:
            cscl = [self.getCanonical(obj, self.canonical_user) for obj in cobj]
            for (_i, _c) in enumerate(cscl):
                if len(cobj) > 1:
                    for _k in ['ty','tx','tz']:
                        config['c' + _k ][_i] = _c[_k]
                    config['s'][_i] *= _c['s']
                else:
                    for _k in ['ty', 'tx', 'tz']:
                        config['c' + _k] = _c[_k]
                    config['s'] *= _c['s']

        else:
            for _k in ['cty', 'ctx', 'ctz']:
                if _k not in config:
                    config[_k] = [0] * cobj if len(cobj) > 1 else 0
        config.setdefault('internal_canonical', self.internal_canonical)
        return irr(config)

    def get_images(self, preproc, get_models=False):
        if get_models:
            self.get_models()
        name = self.specific_name + '_' + get_image_id(preproc)
        cachedir = self.cache_home()
        meta = self.meta
        if self.noise:   
            preproc = copy.deepcopy(preproc)
            preproc['noise'] = self.noise
            meta = meta.addcols(range(len(meta)), names=['noise_seed'])
        window_type = 'texture'
        preproc = copy.deepcopy(preproc)
        preproc['size'] = tuple(preproc['size'])
        size = preproc['size']
        irr = self.imager.get_map(preproc, window_type)
        image_map = larray.lmap(irr, meta)	
        return larray.cache_memmap(image_map, name=name, basedir=cachedir)

    def saveCanonical(self, preproc, config):
        """Checks to see that config contains necessary fields. 
        Increments all prior versions of obj
        submitted by user by 1, and adds new database entry with version = 0. 
        Returns rendered image.
        """
        keys_passed = config.keys()
        if ('user' in keys_passed) and ('obj' in keys_passed):
            self.col.update({'obj': config['obj'],
                             'user': config['user']},
                             {'$inc': {'version': 1}}, multi=True, safe=True)
            config['version'] = 0
            self.col.insert(config, safe=True)
            return self.get_image(preproc, config)
        else:
            raise Exception('Parameters must include "user" and "obj" fields')

    def getCanonical(self, obj, user=None, version=0):
        #Returns most recent database entry to match query, if it exists.
        if user is None:
            user = self.canonical_user
        return self.col.find_one({'obj': obj,
                                  'user': user,
                                  'version': version})

        
        
class CanonicalBase(GenerativeBase):
	#Subclass to save canonical scale/pose in a database.
	#Runs get_image and saves parameters in database each time.

    def __init__(self, dbname='canonicalviews', colname='handset', hostname='localhost', port=27017):
        GenerativeBase.__init__(self, data=None)
        self.dbname = dbname
        self.colname = colname
        self.hostname = hostname
        self.port = port
        self.conn = pymongo.Connection(port=self.port, host=self.hostname) #Open connection
        self.db = self.conn[self.dbname]
        self.col = self.db[self.colname]
        self.col.ensure_index([('obj', pymongo.ASCENDING),
                                 ('user', pymongo.ASCENDING),
                                 ('version', pymongo.DESCENDING)],
                              unique=True)

    def saveCanonical(self, preproc, config):
        """Checks to see that config contains necessary fields. 
        Increments all prior versions of obj
        submitted by user by 1, and adds new database entry with version = 0. 
        Returns rendered image.
        """
        keys_passed = config.keys()
        if ('user' in keys_passed) and ('obj' in keys_passed):
            self.col.update({'obj': config['obj'],
                             'user': config['user']},
                             {'$inc': {'version': 1}}, multi=True, safe=True)
            config['version'] = 0
            self.col.insert(config, safe=True)
            return self.get_image(preproc, config)
        else:
            raise Exception('Parameters must include "user" and "obj" fields')

    def getCanonical(self, obj, user, version=0):
        #Returns most recent database entry to match query, if it exists.
        return self.col.find_one({'obj': obj,
                                  'user': user,
                                  'version': version})


class GenerativeDatasetBase(GenerativeBase):
    """A class that generates randomly sampled metadata for single objects 
    from a set of templates.  Datasets are implemented as subclasses of this 
    class which define the "templates" attribute 
    as class attributes
    """
    model_categories = dict_inverse(model_info.MODEL_CATEGORIES)
    #model_categories.update(dict_inverse(model_info.MODEL_CATEGORIES2))
    
    def _get_meta(self):
        #generate params 
        models = self.models
        templates = self.templates
        use_canonical = self.use_canonical
        internal_canonical = self.internal_canonical
        
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
                           float(l.get('tx', 0)),
                           float(l['s']),
                           tname,
                           l['id'],
                           l.get('texture'),
                           l.get('texture_mode'))
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
                                                     'tx',
                                                     's',
                                                     'tname',
                                                     'id', 
                                                     'texture',
                                                     'texture_mode'])
        if use_canonical:
            meta = meta.addcols([np.zeros((len(meta),)) for _ in range(3)], names=['cty','ctz','ctx'])
            objs = np.unique(meta['obj'])
            cscl = dict([(obj, self.getCanonical(obj, self.canonical_user)) for obj in objs])
            for obj in objs:
                for _k in ['ty','tz','tx']:
                    meta['c' + _k] -= cscl[obj][_k]
                meta['s'][meta['obj'] == obj] *= cscl[obj]['s']
        if internal_canonical:
            meta = meta.addcols([np.ones((len(meta),))], names = ['internal_canonical'])
                
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
    utestcategorylist = [np.unique(categories[_t]) for _t in test_ind_list]
    utraincategories = np.unique(categories[train_inds])    
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
                if cat in utraincategories:
                    assert len(cat_train_inds) >= npc_train, ( 
                                    'not enough train for %s, %d, %d' % (cat,
                                                len(cat_train_inds), npc_train))
                cat_train_inds.sort()
                p = rng.permutation(len(cat_train_inds))
                cat_train_inds_split = cat_train_inds[p[:npc_train]]
                splits[split_ind]['train'] += cat_train_inds_split.tolist()
                for _ind, (test_inds, utc) in enumerate(zip(test_ind_list, utestcategorylist)):
                    npc_test = npc_tests[_ind]
                    cat_test_inds = test_inds[categories[test_inds] == cat]
                    cat_test_inds_c = np.array(list(
                             set(cat_test_inds).difference(
                             cat_train_inds_split).difference(cat_validates)))
                    if cat in utc:
                        assert len(cat_test_inds_c) >= npc_test, (
                                          'not enough test for %s %d %d' % 
                                      (cat, len(cat_test_inds_c), npc_test))
                    p = rng.permutation(len(cat_test_inds_c))
                    cat_test_inds_split = cat_test_inds_c[p[: npc_test]]
                    name = test_names[_ind]
                    splits[split_ind][name] += cat_test_inds_split.tolist()
            else:
                all_cat_test_inds = []
                for _ind, (test_inds, utc) in enumerate(zip(test_ind_list, utestcategorylist)):
                    npc_test = npc_tests[_ind]
                    cat_test_inds = test_inds[categories[test_inds] == cat]
                    cat_test_inds_c = np.sort(np.array(list(
                             set(cat_test_inds).difference(cat_validates))))
                    if cat in utc:
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
                if cat in utraincategories:
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
    
    def __init__(self, data=None, **kwargs):
        GenerativeDatasetBase.__init__(self, data, **kwargs)
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


class GenerativeDatasetAllCategory1(GenerativeDataset4):
    models = list(itertools.chain(*model_info.MODEL_CATEGORIES.values()))


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

    HUMAN_DATA = [('dataset5_category.json', 'category', {'labelfunc': 'category',
                         'split_by': 'obj',
                         'test_q': {},
                         'train_q': {}},
                    'e34830556a6f883de469e79106221945d6fc3446')]


class GenerativeDatasetAllCategory1Mid(GenerativeDataset5):
    models = list(itertools.chain(*[v for k, v in model_info.MODEL_CATEGORIES.items() if k != 'plants']))
    templates = GenerativeDataset5.templates
    templates[0]['n_ex_per_model'] = 100


class GenerativeDatasetAllCategory1Mid2(GenerativeDataset5):
    models = list(itertools.chain(*[v for k, v in model_info.MODEL_CATEGORIES.items() if k != 'plants']))
    templates = GenerativeDataset5.templates
    templates[0]['n_ex_per_model'] = 200


class GenerativeDataset5NewSurfaces(GenerativeDataset4):   
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
                     'texture': choice(model_info.SURFACES),
                     'texture_mode': choice([4, 6])
                     }
                  }]   

    HUMAN_DATA = [('dataset5NewSurfaces_category.json', 'category', {'labelfunc': 'category',
                         'split_by': 'obj',
                         'test_q': {},
                         'train_q': {}}, '17a417e2bb41789100b70c17816d2835d7156c9a')]


class GenerativeDatasetCategories2(GenerativeDataset4):   
    models = list(itertools.chain(*model_info.MODEL_CATEGORIES2.values()))
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
                     'texture': None,
                     'texture_mode': None
                     }
                  }]   


class GenerativeDatasetCategories2SurfaceReplacement(GenerativeDataset4):   
    models = list(itertools.chain(*model_info.MODEL_CATEGORIES2.values()))
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
                     's': uniform(2./3, 3),
                     'ty': uniform(-0.5, 0.5),
                     'tz': uniform(-0.5, 0.5),
                     'ryz': uniform(-180., 180.),
                     'rxy': uniform(-180., 180.),
                     'rxz': uniform(-180., 180.),
                     'texture': choice(model_info.SURFACES2),
                     'texture_mode': 4
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
        model_categories.update(dict_inverse(model_info.MODEL_CATEGORIES2))
        
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
                  

def parse_raw_human_data(resultfile, ldict, blocks):
    X = open(resultfile, 'rU').read().strip('\n').split('\n')
    T = [x.split('\t') for x in X]
    fields = T[0]

    records = []
    R = []
    for t in T[1:]:
        r = [json.loads(tt) if ind != 4 else tt for ind, tt in enumerate(t[:-1]) ]
        Z = json.loads(t[-1].replace('""','"')[1:-1])[0]
        ss = map(urllib.url2pathname, map(str, Z['StimShown']))
        if ldict:
            resp = np.array([ldict[rs] for rs in Z['Response']])
        else:
            resp = np.array(Z['Response'])
        Z['blocks'] = {}
        for bn, binds in blocks:
            Z['blocks'][bn] = {}
            Z['blocks'][bn]['StimShown'] = ss[binds[0]: binds[1]]
            Z['blocks'][bn]['Response'] = resp[binds[0]: binds[1]]
        r.append(Z)
        R.append(Z)
        records.append(r)

    return R, fields, records
                  
                  
def parse_human_data(human_data):
    blocks = [('', (None, None))]
    data = {}
    for (resultfile, name, task, sha1) in human_data:
        R, fields, records = parse_raw_human_data(resultfile, None, blocks)
        data[name] = {'task': task,
                      'data': [r['blocks'][''] for r in R]}
                
    return data


def isstring(x):
    try:
        x + ''
    except TypeError:
        return False
    else:
        return True
