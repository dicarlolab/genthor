import os
import sys
import bpy
import mathutils
import pdb


"""
Usage: 
$ blender -b -P obj2egg.py -- <.obj filename>

"""


def import_obj(pth):
    bpy.ops.import_scene.obj(filepath=pth)


def export_egg(pth):
    import io_scene_egg.yabee_libs.egg_writer
    #from io_scene_egg.yabee_libs import egg_writer
    print("RELOADING MODULES")
    import imp
    imp.reload(io_scene_egg.yabee_libs.egg_writer)

    #: { "animation_name" : (start_frame, end_frame, frame_rate) }
    ANIMATIONS = {"anim1":(0,10,5), }
    #: "True" to interprete an image in the uv layer as the texture
    EXPORT_UV_IMAGE_AS_TEXTURE = False 
    #: "True" to copy texture images together with main.egg
    COPY_TEX_FILES = True
    #: Path for the copied textures. Relative to the main EGG file dir.
    #: For example if main file path is "/home/username/test/test.egg",
    #: texture path is "./tex", then the actual texture path is 
    #: "/home/username/test/tex"
    TEX_PATH = "./tex" #os.path.join(os.path.split(pth)[0], "tex")
    #: "True" to write an animation data into the separate files
    SEPARATE_ANIM_FILE = False #True
    #: "True" to write only animation data
    ANIM_ONLY = False
    #: number of sign after point
    FLOATING_POINT_ACCURACY = 6
    #: Enable tangent space calculation. Tangent space needed for some 
    # shaders/autoshaders, but increase exporting time
    # "NO", "INTERNAL", "PANDA"
    # "INTERNAL" - use internal TBS calculation
    # "PANDA" - use egg-trans to calculate TBS
    # "NO" - do not calc TBS
    CALC_TBS = "NO" #"PANDA"#
    #: Type of texture processing. May be "SIMPLE" or "BAKE".
    # "SIMPLE" - export all texture layers as MODULATE. 
    # Exceptions: 
    #   use map normal == NORMAL
    #   use map specular == GLOSS
    #   use map emit == GLOW
    # "BAKE" - bake textures. BAKE_LAYERS setting up what will be baked.
    # Also diffuse color of the material would set to (1,1,1) in the 
    # "BAKE" mode
    #TEXTURE_PROCESSOR = "BAKE"
    TEXTURE_PROCESSOR = "SIMPLE"
    # type: (size, do_bake)
    BAKE_LAYERS = {"diffuse":(512, True),
                   "normal":(512, True),
                   "gloss": (512, True),    # specular
                   "glow": (512, False)      # emission
                   }

    egg_writer = io_scene_egg.yabee_libs.egg_writer
    egg_writer.write_out(pth, 
                        ANIMATIONS,
                        EXPORT_UV_IMAGE_AS_TEXTURE, 
                        SEPARATE_ANIM_FILE, 
                        ANIM_ONLY,
                        COPY_TEX_FILES, 
                        TEX_PATH, 
                        FLOATING_POINT_ACCURACY,
                        CALC_TBS,
                        TEXTURE_PROCESSOR,
                        BAKE_LAYERS)

    #bpy.ops.wm.addon_enable(module="io_scene_egg")
    #bpy.ops.export.panda3d_egg(pth)


def transform_model(rot):
    
    ## Select the meshes, add them to active object
    bpy.ops.object.select_by_type(type="MESH")

    ## Center model
    #not sure how, not necessary at the moment
    # bpy.ops.view3d.snap_cursor_to_center()
    # bpy.ops.view3d.snap_selected_to_cursor()

    ## Compute scale and location
    # Compute bounding box of selection in abs coords
    BB0 = [0., 0., 0.]
    BB1 = [0., 0., 0.]
    for obj in bpy.context.selected_objects:
        for i in range(3):
            BB0[i] = min(BB0[i], obj.location[i] - obj.dimensions[i] / 2.)
            BB1[i] = max(BB1[i], obj.location[i] + obj.dimensions[i] / 2.)
    Dim = [bb1 - bb0 for bb0, bb1 in zip(BB0, BB1)]
    Loc = [(bb1 + bb0) / 2. for bb0, bb1 in zip(BB0, BB1)]

    ## Re-scale
    scale = [1. / min(Dim)] * 3
    bpy.ops.transform.resize(value=scale)

    ## Re-locate
    bpy.ops.transform.translate(value=(-Loc[0], -Loc[1], -Loc[2]))

    ## Rotate
    bpy.ops.transform.rotate(value=(rot[0],), axis=(0., 0., 1.))
    bpy.ops.transform.rotate(value=(rot[1],), axis=(0., 1., 0.))
    bpy.ops.transform.rotate(value=(rot[2],), axis=(1., 0., 0.))

    # # Swap coords
    # bpy.ops.transform.rotate(value=(90.,), axis=(0., 1., 0.))
    # bpy.ops.transform.rotate(value=(90.,), axis=(1., 0., 0.))


def run(obj_path, egg_path, rot):
    # Empty the current scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    
    # Import obj into scene
    import_obj(obj_path)

    # Do whatever you need to do within the Blender scene
    transform_model(rot)

    # Export egg
    #outpth = os.path.splitext(pth)[0] + ".egg"
    export_egg(egg_path)

    # # Drop to debugger
    # print("\nYou're now in the debugger, within the Blender context\n")
    # pdb.set_trace()



# Main
if __name__ == "__main__":
    # Get command line arguments
    args = sys.argv
    
    # Get the relevant arguments, which follow the "--" sign
    pyargs = args[args.index("--") + 1:]

    # Get the .obj filename
    obj_path = pyargs[0]

    # Put together output egg path
    if len(pyargs) < 2:
        egg_path = os.path.splitext(obj_path)[0] + ".egg"
    else:
    # Get the .egg filename from cmd line
        egg_path = pyargs[1]

    # Put together rotation
    if len(pyargs) < 3:
        rot = [0., 0., 0.]
    else:
        # Get rotation from cmd line
        param_str = pyargs[2]
        rot = [float(s) for s in param_str.split(",")]

    # Run the import
    run(obj_path, egg_path, rot)

