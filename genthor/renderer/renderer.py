#!/usr/bin/env python

import os
import sys
import numpy as np
from pandac.PandaModules import CullFaceAttrib
from pandac.PandaModules import TexGenAttrib
from pandac.PandaModules import TextureStage
from pandac.PandaModules import NodePath
import genthor as gt
import genthor.tools as tools
from genthor.renderer.lightbase import LightBase
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


def construct_scene(lbase, modelpth, bgpath, scale, pos, hpr, bgscale, bghp,
                    scene=None):
    """ Constructs the scene per the parameters. """

    # Default scene is lbase's rootnode
    if scene is None:
        scene = lbase.rootnode
    
    # Modelpth points to the model .egg/.bam file
    if isinstance(modelpth, str):
        modelpths = [modelpth]
        scales = [scale]
        poses = [pos]
        hprs = [hpr]
    else:  
        modelpths = modelpth
        scales = scale
        poses = pos
        hprs = hpr
    assert hasattr(modelpths, '__iter__')
    assert hasattr(scales, '__iter__')
    assert hasattr(poses, '__iter__')
    assert hasattr(hprs, '__iter__')
    assert len(modelpths) == len(scales) == len(hprs) == len(poses)
        
    objnodes = []
    for mpth, scale, hpr, pos in zip(modelpths, scales, hprs, poses):
        objnode = tools.read_file(lbase.loader.loadModel, mpth)
        objnode.setScale(scale[0], scale[0], scale[0])
        #objnode.setPos(pos[0], pos[1], 0.)
        objnode.setPos(pos[0], 0., pos[1])
        objnode.setHpr(hpr[0], hpr[1], hpr[2])
        objnode.setTwoSided(1)
        objnodes.append(objnode)

    #XXX:  What is this code doing? it's been disabled, I guess
    #should it be removed?  is there a todo reminder here?
    # Environment map
    if bgpath and False:
        envtex = tools.read_file(lbase.loader.loadTexture, bgpath)
        # Map onto object
        ts = TextureStage('env')
        ts.setMode(TextureStage.MBlendColorScale)
        objnode.setTexGen(ts, TexGenAttrib.MEyeSphereMap)
        objnode.setTexture(ts, envtex)

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

    # Reparent to a single scene node
    for objnode in objnodes:
        objnode.reparentTo(scene)
    bgnode.reparentTo(scene)

    return objnodes, bgnode


def model_name2path(modelname):
    """ Take model name 'modelname' and return the path to it."""

    modelpath = os.path.join(gt.MODEL_PATH, modelname, modelname + ".bam")
    return modelpath


def bg_name2path(bgname):
    """ Take background name 'bgname' and return the path to it."""
    
    bgpath = os.path.join(gt.BACKGROUND_PATH, bgname)
    return bgpath


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
    objnode, bgnode = construct_scene(lbase, modelpath, bgpath, scale, pos, hpr,
                                      bgscale, bghp)

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
        model_name2path("bloodhound"), #MB26897"),
        bg_name2path("DH214SN.jpg"), #DH201SN.jpg"),
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
