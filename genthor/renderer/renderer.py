#!/usr/bin/env python

import genthor as gt
import genthor.modeltools.convert_models as cm
import genthor.modeltools.tools as mt
from genthor.renderer.lightbase import LightBase
import genthor.tools as tools
from libpanda import Point3
import numpy as np
import os
from pandac.PandaModules import CullFaceAttrib
from pandac.PandaModules import NodePath
from pandac.PandaModules import TexGenAttrib
from pandac.PandaModules import TextureStage
from pandac.PandaModules import Texture
from panda3d.core import *
from direct.gui.OnscreenImage import OnscreenImage
import sys
import pdb


def setup_renderer(window_type, size=(256, 256), light_spec=None, cam_spec=None):
    """ Sets up the LightBase rendering stuff."""

    # Initialize
    lbase = LightBase()
    rootnode = lbase.rootnode

    if window_type == "onscreen":
        output = lbase.make_window(size, "window")
    elif window_type == "offscreen":
        output = lbase.make_buffer(size, "buffer")
    elif window_type == "texture":
        output, tex = lbase.make_texture_buffer(size, "texturebuffer",
                                                mode='RTMCopyRam')
    else:
        raise ValueError("Unknown window type: %s" % window_type)

    # Clear out frame contents
    lbase.render_frame()
    lbase.render_frame()

    # Set up a camera
    if cam_spec is None:
        cam_spec = {}
    print('cam_spec', cam_spec)
    cam_x, cam_z, cam_y = cam_spec.get('cam_pos', (0., -20., 0.))
    scene_width = cam_spec.get('scene_width', 3.)
    fov = 2. * np.degrees(np.arctan(scene_width / (2. * np.abs(cam_z))))
    cam_lx, cam_lz, cam_ly = cam_spec.get('cam_lookat', (0., 0., 0.)) 
    cam_rot = cam_spec.get('cam_rot', 0.)
    camera = lbase.make_camera(output)
    lens = camera.node().getLens()
    lens.setMinFov(fov)    
    # Position the camera
    camera_rot = rootnode.attachNewNode('camera_rot')
    lbase.cameras.reparentTo(camera_rot)
    lbase.cameras.setPos(cam_x, cam_z, cam_y)
    lbase.cameras.lookAt(cam_lx, cam_lz, cam_ly)
    camera_rot.setH(cam_rot)
    
    # Lights
    lights = LightBase.make_lights(light_spec=light_spec)
    lights.reparentTo(rootnode)
    for light in lights.getChildren():
        rootnode.setLight(light)

    # # Pause while setup finishes
    # time.sleep(0.02)

    return lbase, output


def construct_scene(lbase, modelpath, bgpath, scale, pos, hpr, 
                    bgscale, bghp,
                    texture=None,
                    internal_canonical=False,
                    check_penetration=False, 
                    light_spec=None, 
                    use_envmap=False):
    """ Constructs the scene per the parameters. """

    # Default scene is lbase's rootnode
    if bgpath is not None:
        bgpath = mt.resolve_bg_path(bgpath)
    rootnode = lbase.rootnode
    # Modelpath points to the model .egg/.bam file

    if isinstance(modelpath, str):
        modelpaths = [modelpath]
        scales = [scale]
        poses = [pos]
        hprs = [hpr]
        textures = [texture]
    else:  
        modelpaths = modelpath
        scales = scale
        poses = pos
        hprs = hpr
        textures = texture

    texmodes = []
    for _i, _t in enumerate(textures):
        if isinstance(_t, tuple):
            texfile, texmode = _t
            texmodes.append(texmode)
            textures[_i] = texfile
        else:
            texmodes.append(TexGenAttrib.MWorldNormal)    

    assert hasattr(modelpaths, '__iter__')
    assert hasattr(scales, '__iter__')
    assert hasattr(poses, '__iter__')
    assert hasattr(hprs, '__iter__')
    assert hasattr(textures, '__iter__')
    assert len(modelpaths) == len(scales) == len(hprs) == len(poses) == len(textures), (len(modelpaths), len(scales), len(hprs), len(poses), len(textures))
        
    modelpaths = map(mt.resolve_model_path, modelpaths)
    modelpaths = map(cm.autogen_egg, modelpaths)
    textures = map(mt.resolve_texture_path, textures)
    objnodes = []
    for mpth, scale, hpr, pos, t, tm in zip(modelpaths, scales, hprs, poses, textures, texmodes):
        objnode = tools.read_file(lbase.loader.loadModel, mpth)
        if t is not None: 
            #ts = TextureStage('ts')
            ts = TextureStage.get_default()
            ts.setMode(TextureStage.MReplace) 
            tex = tools.read_file(lbase.loader.loadTexture, t) 
            objnode.setTexGen(ts, tm)
            objnode.setTexture(tex, 6)
        
        robjnode = rootnode.attachNewNode('root_' + objnode.get_name())
        objnode.reparentTo(robjnode)
        if internal_canonical:
            vertices = np.array(objnode.getTightBounds())
            initial_scale_factor = max(abs(vertices[0]-vertices[1]))
            cscale = 1.2/initial_scale_factor
            ppos = vertices.mean(0) * cscale
            objnode.setPos(-ppos[0], -ppos[1], -ppos[2])
            objnode.setScale(cscale, cscale, cscale)
            
        robjnode.setScale(scale[0], scale[0], scale[0])
        robjnode.setPos(pos[0], -pos[2], pos[1])
        robjnode.setHpr(hpr[2], hpr[1], hpr[0])
        robjnode.setTwoSided(1)

        objnodes.append(robjnode)

    if check_penetration:
        for (i, n1) in enumerate(objnodes):
            for j, n2 in enumerate(objnodes[i+1:]):
                p = is_penetrating(n1, n2)
                if p:
                    for onode in objnodes:
                        onode.removeNode()
                    raise PenetrationError(i, j, n1, n2)

    # Environment map
    if bgpath and use_envmap:
        envtex = tools.read_file(lbase.loader.loadTexture, bgpath)
        # Map onto object
        ts = TextureStage('env')
        ts.setMode(TextureStage.MBlendColorScale)
        if not isinstance(use_envmap, list):
            use_envmap = [use_envmap] * len(objnodes)
        for _objnode, ue in zip(objnodes, use_envmap):
            if ue:
                if isinstance(ue, str):
                    envtex0 = tools.read_file(lbase.loader.loadTexture, mt.resolve_texture_path(ue))
                else:
                    envtex0 = envtex
                _objnode.setTexGen(ts, TexGenAttrib.MEyeSphereMap)
                _objnode.setTexture(ts, envtex0)

    if bgpath and not np.isinf(bghp[0]):
        bgtex = tools.read_file(lbase.loader.loadTexture, bgpath)
        # Set as background
        #plane = cm.autogen_egg(mt.resolve_model_path('plane2d'))
        bgnode = lbase.loader.loadModel('smiley')
        # Get material list
        bgnode.clearMaterial()
        bgnode.clearTexture()
        bgnode.setAttrib(CullFaceAttrib.make(
            CullFaceAttrib.MCullCounterClockwise))
        bgnode.setTexture(bgtex, 2)
        c = 5.
        bgnode.setScale(c * bgscale[0], c * bgscale[0], c * bgscale[0])
        bgnode.setPos(0, 0, 0) #0)
        bgnode.setHpr(bghp[0], bghp[1], 0.) 
        # Detach point light
        plight1 = lbase.rootnode.find('**/plight1')
        if plight1:
            plight1.detachNode()
    elif bgpath:
        bgnode = NodePath("empty-bgnode")
        imageObject = OnscreenImage(image = bgpath, pos = (0, 0, 0), scale=tuple(bgscale), parent=bgnode, base=lbase)
    else:
        bgnode = NodePath("empty-bgnode")
    bgnode.reparentTo(rootnode)

    if light_spec is not None:
        lbase.rootnode.clearLight()
        lights = LightBase.make_lights(light_spec=light_spec, lname='scene_lights')
        lights.reparentTo(rootnode)
        for light in lights.getChildren():
            rootnode.setLight(light)

    return objnodes, bgnode


def is_penetrating(node0, node1):
    """ Tests whether two nodes' geometries are overlapping by
    comparing their axis-aligned bounding boxes (AABB)."""
    # bb0 = [Vec3(0., 0., 0.), Vec3(5., 5., 5.)]
    # bb1 = [Vec3(1., 1., 1.), Vec3(16., 16., 16.)]

    # Allocate Vec3 storage
    bb0 = [Point3(0., 0., 0.), Point3(0., 0., 0.)]
    bb1 = [Point3(0., 0., 0.), Point3(0., 0., 0.)]

    # Get tight AABBs
    node0.calcTightBounds(bb0[0], bb0[1])
    node1.calcTightBounds(bb1[0], bb1[1])

    # Perform the test
    BB0 = np.array(bb0)
    BB1 = np.array(bb1)
    f_penetrating = not (np.any(BB0[0] > BB1[1]) or np.any(BB1[0] > BB0[1]))

    return f_penetrating


def run(args):
    """ run() is called by the command line interface. It can serve as
    a template for operating the contained functions from within a
    Python module."""

    window_type = "onscreen"

    # Set up the renderer
    lbase, output = setup_renderer(window_type)
    if window_type == "offscreen":
        tex = output.getTexture()
    
    # Construct a scene
    modelpath, bgpath, scale, pos, hpr, bgscale, bghp = args
    objnode, bgnode = construct_scene(lbase, modelpath, bgpath,
                                      scale, pos, hpr, bgscale, bghp)

    # Render the scene
    lbase.render_frame()

    if window_type == "offscreen":
        # Get the image
        img = lbase.get_tex_image(tex)
    else:
        img = None

    return lbase, img


if __name__ == "__main__":
    # Command line usage
    args = sys.argv[1:]

    # Defaults
    # args = (modelpath, bgpath, scale, pos, hpr, bgscale, bghp)
    args = [
        mt.resolve_model_path("bloodhound"), #MB26897"),
        mt.resolve_bg_path("DH214SN.jpg"), #DH201SN.jpg"),
        # os.path.join(os.environ["HOME"],
        #              "Dropbox/genthor/rendering/backgrounds/Hires_pano.jpg"),
        (1.,),
        (0., 0.),
        (0., 0., 0.),
        (1.,),
        (0., 0.),
        ]
    #args[1] = ""
    
    lbase = run(args)

    raw_input("press ENTER to exit...")


class PenetrationError(Exception):
    def __init__(self, i, j, n1, n2):
        self.msg = 'Nodes %d (%s) and %d (%s) are penetrating' % (i, repr(n1), i+1+j, repr(n2))
        print(self.msg)
