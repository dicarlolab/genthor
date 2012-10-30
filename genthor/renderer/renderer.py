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
import sys
import pdb


def setup_renderer(window_type, size=(256, 256)):
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
    scene_width = 3.
    cam_z = -20.
    fov = 2. * np.degrees(np.arctan(scene_width / (2. * np.abs(cam_z))))
    camera = lbase.make_camera(output)
    lens = camera.node().getLens()
    lens.setMinFov(fov)
    # Position the camera
    camera_rot = rootnode.attachNewNode('camera_rot')
    lbase.cameras.reparentTo(camera_rot)
    lbase.cameras.setPos(0, cam_z, 0)
    lbase.cameras.lookAt(0, 0, 0)
    camera_rot.setH(0.)
    # Lights
    lights = LightBase.make_lights()
    lights.reparentTo(rootnode)
    for light in lights.getChildren():
        rootnode.setLight(light)

    # # Pause while setup finishes
    # time.sleep(0.02)

    return lbase, output


def construct_scene(lbase, modelpath, bgpath, scale, pos, hpr, 
                    bgscale, bghp,
                    texture=None,
                    scene=None, check_penetration=False):
    """ Constructs the scene per the parameters. """

    # Default scene is lbase's rootnode
    if scene is None:
        scene = lbase.rootnode

    bgpath = mt.resolve_bg_path(bgpath)
    
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

    assert hasattr(modelpaths, '__iter__')
    assert hasattr(scales, '__iter__')
    assert hasattr(poses, '__iter__')
    assert hasattr(hprs, '__iter__')
    assert hasattr(textures, '__iter__')
    assert len(modelpaths) == len(scales) == len(hprs) == len(poses) == len(textures)
        
    modelpaths = map(mt.resolve_model_path, modelpaths)
    modelpaths = map(cm.autogen_egg, modelpaths)
    textures = map(mt.resolve_texture_path, textures)
    objnodes = []
    for mpth, scale, hpr, pos, t in zip(modelpaths, scales, hprs, poses, textures):
        objnode = tools.read_file(lbase.loader.loadModel, mpth)
        if t is not None: 
            ts = TextureStage('ts')
            ts.setMode(TextureStage.MReplace) 
            tex = tools.read_file(lbase.loader.loadTexture, t) 
            objnode.setTexGen(ts, TexGenAttrib.MWorldNormal)
            #tex.setWrapU(Texture.WMMirror)
            #tex.setWrapV(Texture.WMMirror)
            #objnode.setTexProjector(TextureStage.getDefault(), scene, objnode)
            objnode.setTexture(tex, 1)
            
        objnode.setScale(scale[0], scale[0], scale[0])
        objnode.setPos(pos[0], 0., pos[1])
        objnode.setHpr(hpr[2], hpr[1], hpr[0])
        objnode.setTwoSided(1)
        objnodes.append(objnode)

    # Environment map
    #if bgpath and False:
    #    envtex = tools.read_file(lbase.loader.loadTexture, bgpath)
    #    # Map onto object
    #    ts = TextureStage('env')
    #    ts.setMode(TextureStage.MBlendColorScale)
    #    objnode.setTexGen(ts, TexGenAttrib.MEyeSphereMap)
    #    objnode.setTexture(ts, envtex)

    if bgpath:
        bgtex = tools.read_file(lbase.loader.loadTexture, bgpath)
        # Set as background
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
    else:
        bgnode = NodePath("empty-bgnode")

    if check_penetration:
        for (i, n1) in enumerate(objnodes):
            for j, n2 in enumerate(objnodes[i+1:]):
                p = is_penetrating(n1, n2)
                assert not p, 'Nodes %d (%s) and %d (%s) are penetrating' % (i, repr(n1), i+1+j, repr(n2))
                    
    # Reparent to a single scene node
    for objnode in objnodes:
        objnode.reparentTo(scene)
    bgnode.reparentTo(scene)

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

