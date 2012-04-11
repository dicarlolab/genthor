#!/usr/bin/env python

import os
import sys
import numpy as np
from pandac.PandaModules import CullFaceAttrib
from pandac.PandaModules import TexGenAttrib
from pandac.PandaModules import TextureStage
from pandac.PandaModules import NodePath
import genthor as gt
from lightbase import LightBase
import pdb


def setup_renderer(window_type, size=(512, 512)):
    """ Sets up the LightBase rendering stuff."""

    # Initialize
    lbase = LightBase()
    rootnode = lbase.rootnode

    if window_type == "onscreen":
        output = lbase.make_window(size, "window")
    else:
        output, tex = lbase.make_texture_buffer(size, "buffer",
                                                mode='RTMCopyRam')

    # Set up a camera
    camera = lbase.make_camera(output)
    lens = camera.node().getLens()
    lens.setMinFov(45)
    # Position the camera
    camera_rot = rootnode.attachNewNode('camera_rot')
    lbase.cameras.reparentTo(camera_rot)
    lbase.cameras.setPos(0, -24, 0)
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


def read_file(func, filepth):
    """ Returns func(filepath), first trying absolute path, then
    relative."""

    try:
        out = func(filepth)
    except IOError:
        try:
            out = func(os.path.join(os.getcwd(), filepth))
        except IOError as exc:
            raise exc
    return out


def construct_scene(lbase, modelpth, bgpath, scale, pos, hpr, bgscale, bghp):
    """ Constructs the scene per the parameters. """
    
    # Modelpth points to the model .egg/.bam file
    node = read_file(lbase.loader.loadModel, modelpth)
    node.setScale(scale, scale, scale)
    node.setPos(pos[0], pos[1], 0.)
    node.setHpr(hpr[0], hpr[1], hpr[2])

    # Environment map
    if bgpath:
        envtex = read_file(lbase.loader.loadTexture, bgpath)
        bgtex = envtex.makeCopy()
        # Map onto object
        ts = TextureStage('env')
        ts.setMode(TextureStage.MBlendColorScale)
        node.setTexGen(ts, TexGenAttrib.MEyeSphereMap)
        node.setTexture(ts, envtex)
        # Set as background
        scenenode = lbase.loader.loadModel('smiley')
        # Get material list
        scenenode.clearMaterial()
        scenenode.clearTexture()
        scenenode.setAttrib(CullFaceAttrib.make(
            CullFaceAttrib.MCullCounterClockwise))
        scenenode.setTexture(bgtex, 2)
        scenenode.setPos(0., 0., 0.)
        scenenode.setScale(bgscale, bgscale, bgscale)
        scenenode.setH(bghp[0])
        scenenode.setP(bghp[1])
        # Detach point light
        plight1 = lbase.rootnode.find('**/plight1')
        plight1.detachNode()
    else:
        scenenode = NodePath("empty-scenenode")

    node.reparentTo(lbase.rootnode)
    scenenode.reparentTo(lbase.rootnode)

    return scenenode


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
    scenenode = construct_scene(lbase, modelpath, bgpath, scale, pos, hpr,
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
        model_name2path("reptiles3"),
        #bg_name2path("DH201SN.hdr"),
        os.path.join(os.environ["HOME"],
                     "Dropbox/genthor/rendering/backgrounds/Hires_pano.jpg"),
        (1., 1., 1.),
        (0., 0., 0.),
        (0., 0., 0.),
        (0., 0.),
        (20.),
        ]
    #args[1] = ""
    
    lbase = run(args)
