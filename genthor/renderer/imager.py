#!/usr/bin/env python

from genthor.renderer.lightbase import LightBase
import genthor.renderer.renderer as gr
import Image
import numpy as np
import os

import pdb



class Imager(object):

    lbases = []

    def __init__(self):
        pass

    def get_map(self, args):


        ImgRendererResizer
       

        return irr


class ImgRendererResizer(object):
    def __init__(self, model_root, bg_root, preproc, lbase, output):
        size = tuple(preproc['size'])
        self.transpose = preproc.get('transpose', False)
        if self.transpose:
            self._shape = tuple(np.array(size)[list(self.transpose)])
        else:
            self._shape = size
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
        tex = self.output.getTexture()
        im = Image.fromarray(self.lbase.get_tex_image(tex))
        if im.mode != self.mode:
            im = im.convert(self.mode)
        rval = np.asarray(im, self._dtype)
        if self.normalize:
            rval -= rval.mean()
            rval /= max(rval.std(), 1e-3)
        else:
            if 'float' in str(self._dtype):
                rval /= 255.0
        if self.transpose:
            rval = rval.transpose(*tuple(self.transpose))
        assert rval.shape[:2] == self._shape[:2], (rval.shape, self._shape)
        return rval

