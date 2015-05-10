#!/usr/bin/env python
""" Contains Imager and ImgRendererResizer class definitions."""
from genthor.renderer.lightbase import LightBase
import genthor.renderer.renderer as gr
from PIL import Image
import numpy as np
import os
import pdb
import scipy.ndimage as ndimage

class Imager(object):
    """ Manages renderers and can produce ImgRenderResizer instances."""

    # A dict that contains all renderers. The keys are
    # tuple(window_type, size), the values are tuple(LightBase
    # instance, output).
    renderers = {}

    def __init__(self, model_root="", bg_root="", check_penetration=False):
        self.model_root = model_root
        self.bg_root = bg_root
        self.check_penetration=check_penetration

    def get_renderer(self, window_type, size, light_spec=None, cam_spec=None):
        """ Initializes a new renderer and adds it to the
        Imager.renderers dict."""
        # Create the LightBase instance/output

        import hashlib
        def get_id(l):
            return hashlib.sha1(repr(l)).hexdigest()
        ls_id = get_id(light_spec)
        cs_id = get_id(cam_spec)

        lbase, output = self.renderers.get((window_type, size, ls_id, cs_id),
                                           gr.setup_renderer(window_type, size, 
                                               light_spec=light_spec,
                                               cam_spec=cam_spec))
        # Add to the Imager.renderers
        self.renderers[(window_type, size, ls_id, cs_id)] = lbase, output
        return lbase, output

    def get_map(self, preproc, window_type, light_spec=None, cam_spec=None):
        """ Returns an ImgRendererResizer instance."""
        # Get a valid renderer (new or old)
        size = tuple(preproc["size"])
        lbase, output = self.renderers.get((window_type, size),
                                           self.get_renderer(window_type, size, 
                                           light_spec=light_spec,
                                           cam_spec=cam_spec))
        # Make the irr instance
        irr = ImgRendererResizer(self.model_root, self.bg_root,
                                 preproc, lbase, output,
                                 check_penetration=self.check_penetration)
        return irr


class ImgRendererResizer(object):
    def __init__(self, model_root, bg_root, preproc, lbase, output, check_penetration=False):
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
        self.check_penetration = check_penetration
        self.noise = preproc.get('noise')
        self.shader = preproc.get('shader')
            
    def rval_getattr(self, attr, objs):
        if attr == 'shape' and self._shape is not None:
            return self._shape
        if attr == 'ndim' and self._ndim is not None:
            return self._ndim
        if attr == 'dtype':
            return self._dtype
        raise AttributeError(attr)
    
    def remove(self):
        children = list(self.lbase.rootnode.getChildren())
        for c in children[2:]:
            c.removeNode()

    def __call__(self, m, remove=True):
        try:
            m['obj_path']
        except:
            oattr = 'obj'
        else:
            oattr = 'obj_path'
        if isstring(m[oattr]):
            modelpath = os.path.join(self.model_root, 
                                     *(m[oattr].split('/')))
            scale = [m['s']]
            pos = [m['ty'], m['tz'], m['tx']]
            hpr = [m['ryz'], m['rxz'], m['rxy']]
            texture = (m['texture'], m['texture_mode'])
        else:
            assert hasattr(m[oattr], '__iter__')
            modelpath = [os.path.join(self.model_root, *(mn.split('/'))) for mn in m[oattr]]
            scale = [[ms] for ms in m['s']]
            pos = zip(m['ty'], m['tz'], m['tx'])
            hpr = zip(m['ryz'], m['rxz'], m['rxy'])
            texture = zip(m['texture'], m['texture_mode'])
        internal_canonical = m['internal_canonical']
        try:
            light_spec = m['light_spec']
        except:
            light_spec = None
        if m['bgname'] is not None:
            bgpath = os.path.join(self.bg_root, m['bgname'])
        else:
            bgpath = None
        try:
            use_envmap = m['use_envmap']
        except:
            use_envmap = False            

        bgscale = [m['bgscale']]
        bghp = [m['bgphi'], m['bgpsi']]
        args = (modelpath, bgpath, scale, pos, hpr, bgscale, bghp)
        if self.shader:
            print('Using shader', self.shader)

        try:
            objnodes, bgnode = gr.construct_scene(self.lbase, 
                                              *args,
                          check_penetration=self.check_penetration,
                          texture=texture,
                          internal_canonical=internal_canonical,
                          light_spec=light_spec, 
                          use_envmap=use_envmap,
                          shader=self.shader)

            self.lbase.render_frame()
        except Exception, e:
            if remove:
                self.remove()
            raise e
        if remove: 
            self.remove()
        tex = self.output.getTexture()
        _arr = self.lbase.get_tex_image(tex)
        im = Image.fromarray(_arr)
        if im.mode != self.mode:
            im = im.convert(self.mode)
        rval = np.asarray(im, self._dtype)
        if self.noise:
            noise = self.noise['magnitude'] * np.random.RandomState(seed=m['noise_seed']).uniform(size=rval.shape[:2]) * rval.std()
            if rval.ndim == 3:
                noise = noise[:, :, np.newaxis]
            rval = ndimage.gaussian_filter(rval + noise, sigma=self.noise['smoothing']).astype(self._dtype)
            rval = np.maximum(np.minimum(rval, 255.), 0)
        if self.normalize:
            rval = rval - rval.mean()
            rval /= max(rval.std(), 1e-3)
        else:
            if 'float' in str(self._dtype):
                rval /= 255.0
        if self.transpose:
            rval = rval.transpose(*tuple(self.transpose))
        assert rval.shape[:2] == self._shape[:2], (rval.shape, self._shape)
        return rval

def isstring(x):
    try:
        x + ''
    except TypeError:
        return False
    else:
        return True
