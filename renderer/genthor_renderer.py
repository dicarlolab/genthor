#!/usr/bin/env python

import os
import sys
import numpy as np
from pandac.PandaModules import CullFaceAttrib
from pandac.PandaModules import TexGenAttrib
from pandac.PandaModules import TextureStage
from pandac.PandaModules import NodePath
from lightbase import LightBase
import pdb


def setup_renderer(window_type):
    """ Sets up the LightBase rendering stuff."""

    # Initialize
    lbase = LightBase()
    rootnode = lbase.rootnode

    # Make a textureBuffer
    size = (512, 512)

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


def construct_scene(lbase, modelpth, envpth, scale, pos, hpr,
                    phitheta, bgscale):
    """ Constructs the scene per the parameters. """
    
    # Modelpth points to the model .egg/.bam file
    node = read_file(lbase.loader.loadModel, modelpth)
    node.setScale(*scale)
    node.setPos(*pos)
    node.setHpr(*hpr)

    # Environment map
    if envpth:
        envtex = read_file(lbase.loader.loadTexture, envpth)
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
        scenenode.setH(phitheta[0])
        scenenode.setP(phitheta[1])
        scenenode.setScale(bgscale, bgscale, bgscale)
        # Detach point light
        plight1 = lbase.rootnode.find('**/plight1')
        plight1.detachNode()
    else:
        scenenode = NodePath("empty-scenenode")

    node.reparentTo(lbase.rootnode)
    scenenode.reparentTo(lbase.rootnode)

    return scenenode
    


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
    modelpath, envpth, scale, pos, hpr, phitheta, bgscale = args
    scenenode = construct_scene(lbase, modelpath, envpth, scale, pos, hpr,
                                phitheta, bgscale)

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
    #modelpath, envpth, scale, pos, hpr, phitheta, bgscale
    args = [
        os.path.join(os.environ["HOME"],
                     "work/genthor/processed_models/reptiles3/reptiles3.bam"),
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
