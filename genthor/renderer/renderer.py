#!/usr/bin/env python

import genthor as gt
from genthor.modeltools.convert_models import panda_convert
from genthor.renderer.lightbase import LightBase
import genthor.tools as tools
import numpy as np
import os
from pandac.PandaModules import CullFaceAttrib
from pandac.PandaModules import NodePath
from pandac.PandaModules import TexGenAttrib
from pandac.PandaModules import TextureStage
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


def resolve_bg_path(bgpth0):
    """ Finds a valid background path."""
    # Image extensions
    img_exts = (".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".png")
    if not os.path.isfile(bgpth0):
        bgpth = os.path.join(gt.BACKGROUND_PATH, os.path.basename(bgpth0))
        if not os.path.isfile(bgpth):
            raise IOError("Not a valid background file: %s" % bgpth)
    else:
        bgpth = bgpth0
    return bgpth
    

def resolve_model_path(modelpth0, f_force_egg=True):
    """ Finds a valid model path. It will convert an .obj model to an
    .egg if necessary."""
    # Model extensionsv
    panda_exts = (".bam", ".egg")
    valid_exts = panda_exts + (".obj", ".tgz", ".tar.gz", ".tbz2", ".tar.bz2")
    # Find a valid file first
    if os.path.isdir(modelpth0):
        ## modelpth is a directory -- determine what kind of objects
        ## are in it, and check that one is .egg/.bam/.obj
        # Get dir listing
        ld = os.listdir(modelpth0)
        # Add all valid filenames to filepths
        filepths = [fn for fn in ld if gt.splitext2(fn)[1] in valid_exts]
        # Verify modelpths has exactly 1 filename
        if len(filepths) != 1:
            raise IOError("Cannot find valid model file in: %s" % modelpth0)
        # Create modelpth
        modelpth = os.path.join(modelpth0, filepths[0])
    elif not os.path.isfile(modelpth0):
        ## modelpth0 is not a directory or file, which means it might
        ## be a model name. Search for it in the eggs and models
        ## directories.
        # model name
        name = gt.splitext2(os.path.basename(modelpth0))[0]
        # possible file paths, ordered by priority (bam > egg > obj)
        filepths = (os.path.join(gt.BAM_PATH, name, name + ".bam"),
                    os.path.join(gt.EGG_PATH, name, name + ".egg"), 
                    os.path.join(gt.OBJ_PATH, name, name + ".obj"))
        # Look for a valid file path
        for filepth in filepths:
            if os.path.isfile(filepth):
                # Found a good one, save it and break
                modelpth = filepth
                break
            else:
                # Keep looking
                modelpth = None
        if modelpth is None:
            # Error if we can't find a valid file
            raise IOError("Cannot find a valid model name: %s" % name)
    else:
        modelpth = modelpth0

    # modelpth is now a valid file, check its extension and create an
    # .egg if it is not a panda file
    name, ext = gt.splitext2(os.path.basename(modelpth))
    if ext not in panda_exts:
        # modelpth is not a panda3d extension
        # The .egg's path
        pandapth = os.path.join(gt.EGG_PATH, name, name + ".egg")

        if not os.path.isfile(pandapth):
            if f_force_egg:
                # The .egg doesn't exist, so convert the input file
                inout_pth = {modelpth: pandapth}
                panda_convert(inout_pth, ext=".egg")
            else:
                raise IOError(("Found valid non-panda model file, but "
                               "f_force_egg must be True to convert (it is "
                               "False): %s") % modelpth)
        else:
            # The .egg already exists, so just use that
            pass
    else:
        # modelpth is a panda3d file, so just use that
        pandapth = modelpth

    return pandapth


def construct_scene(lbase, modelpath, bgpath, scale, pos, hpr, bgscale, bghp,
                    scene=None):
    """ Constructs the scene per the parameters. """

    # Default scene is lbase's rootnode
    if scene is None:
        scene = lbase.rootnode

    modelpath = resolve_model_path(modelpath)
    bgpath = resolve_bg_path(bgpath)
    
    # Modelpth points to the model .egg/.bam file
    objnode = tools.read_file(lbase.loader.loadModel, modelpath)
    objnode.setScale(scale[0], scale[0], scale[0])
    #objnode.setPos(pos[0], pos[1], 0.)
    objnode.setPos(pos[0], 0., pos[1])
    objnode.setHpr(hpr[0], hpr[1], hpr[2])
    objnode.setTwoSided(1)

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
    objnode.reparentTo(scene)
    bgnode.reparentTo(scene)

    return objnode, bgnode


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
        resolve_model_path("bloodhound"), #MB26897"),
        resolve_bg_path("DH214SN.jpg"), #DH201SN.jpg"),
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

