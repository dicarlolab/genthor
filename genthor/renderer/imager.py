#!/usr/bin/env python
""" Contains Imager and ImgRendererResizer class definitions."""
from genthor.renderer.lightbase import LightBase
import genthor.renderer.renderer as gr
import Image
import numpy as np
import os
import pdb


class Imager(object):
    """ Manages renderers and can produce ImgRenderResizer instances."""

    # A dict that contains all renderers. The keys are
    # tuple(window_type, size), the values are tuple(LightBase
    # instance, output).
    renderers = {}

    def __init__(self, model_root="", bg_root=""):
        self.model_root = model_root
        self.bg_root = bg_root

    def get_renderer(self, window_type, size):
        """ Initializes a new renderer and adds it to the
        Imager.renderers dict."""
        # Create the LightBase instance/output
        lbase, output = self.renderers.get((window_type, size),
                                           gr.setup_renderer(window_type, size))
        # Add to the Imager.renderers
        self.renderers[(window_type, size)] = lbase, output
        return lbase, output

    def get_map(self, preproc, window_type):
        """ Returns an ImgRendererResizer instance."""
        # Get a valid renderer (new or old)
        size = preproc["size"]
        lbase, output = self.renderers.get((window_type, size),
                                           self.get_renderer(window_type, size))
        # Make the irr instance
        irr = ImgRendererResizer(self.model_root, self.bg_root,
                                 preproc, lbase, output)
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
                                 m['obj'])
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

