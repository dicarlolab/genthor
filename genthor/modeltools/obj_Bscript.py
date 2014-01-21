"""
Blender script for normalizing .obj files and converting them to either .obj or
.egg.
"""
import os
import shutil
import sys
try:
    import bpy
    import mathutils
except:
    pass
import pdb


"""
Usage: 
$ blender -b -P obj_Bscript.py -- <.obj filename>
"""


def import_obj(pth):
    # Import the .obj
    bpy.ops.import_scene.obj(filepath=pth)


def export_obj(pth):
    # Export the .obj
    bpy.ops.export_scene.obj(filepath=pth, use_normals=True,
                             keep_vertex_order=True) 

    
def export_egg(pth):
    import io_scene_egg
    try:
        io_scene_egg.register()
    except:
        pass
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
    MERGE_ACTOR_MESH = False
    APPLY_MOD = True
    PVIEW = False

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
                         BAKE_LAYERS,
                         MERGE_ACTOR_MESH,
                         APPLY_MOD,
                         PVIEW)


def transform_model(rot):
    def calc_dim_loc(objs):
        # Compute bounding box of selection in abs coords
        BB0 = [999999., 999999., 999999.]
        BB1 = [-999999., -999999., -999999.]
        for obj in objs:
            bb = obj.bound_box
            for i in range(8):
                vert = obj.matrix_world * mathutils.Vector(bb[i])
                for j in range(3):
                    BB0[j] = min(BB0[j], vert[j])
                    BB1[j] = max(BB1[j], vert[j])
        Dim = [bb1 - bb0 for bb0, bb1 in zip(BB0, BB1)]
        Loc = [(bb1 + bb0) / 2. for bb0, bb1 in zip(BB0, BB1)]

        return Dim, Loc
    
    ## Select the meshes, add them to active object
    bpy.ops.object.select_by_type(type="MESH")

    ## Rotate using input angles
    bpy.ops.transform.rotate(value=(rot[0],), axis=(1., 0., 0.))
    bpy.ops.transform.rotate(value=(rot[1],), axis=(0., 1., 0.))
    bpy.ops.transform.rotate(value=(rot[2],), axis=(0., 0., 1.))

    ## There is something funny about how the transformations get
    ## applied, so I need to scale, then recompute location, then
    ## translate. Ideally, I'd translate, then re-scale, but that
    ## doesn't seem to work.
    
    ## Re-scale
    Dim, Loc = calc_dim_loc(bpy.context.selected_objects)
    scale = [1. / max(Dim)] * 3
    bpy.ops.transform.resize(value=scale)

    ## Re-locate 
    Dim, Loc = calc_dim_loc(bpy.context.selected_objects)
    bpy.ops.transform.translate(value=(-Loc[0], -Loc[1], -Loc[2]))

    # ## Swap coords
    # bpy.ops.transform.rotate(value=(90.,), axis=(0., 1., 0.))
    # bpy.ops.transform.rotate(value=(90.,), axis=(1., 0., 0.))


def run(obj_path, out_path, rot=None):
    # Empty the current scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    try:
        # Import obj into scene
        import_obj(obj_path)

        if rot is not None:
            # Do whatever you need to do within the Blender scene
            transform_model(rot)

        ext = os.path.splitext(out_path)[1]
        if ext == ".egg":
            # Export egg
            try:
                export_egg(out_path)
            except Exception as ce:
                print("Error: %s" % ce)
                print("\nThis error may be due to an incompatible"
                      "Yabee exporter or Blender version "
                      "(tested on Blender 2.63 and Yabee r12 for Blender2.63a)."
                      " See blender.org and code.google.com/p/yabee "
                      "for those versions.")
                    
        elif ext == ".obj":
            # Export obj
            if obj_path == out_path:
                # don't clobber
                raise ValueError("Cannot overwrite the obj files: %s" % out_path)
            # copy textures
            shutil.copytree(os.path.join(os.path.split(obj_path)[0], "tex"),
                            os.path.join(os.path.split(out_path)[0], "tex"))
            # export the obj
            export_obj(out_path)
        else:
            raise ValueError("unsupported output type: %s" % ext)

    except Exception as ce:
        print("Error: %s" % ce)
        print("\nYou're now in the debugger, within the Blender context\n")
        pdb.set_trace()


# Main
if __name__ == "__main__":
    # Get command line arguments
    args = sys.argv
    
    # Get the relevant arguments, which follow the "--" sign
    pyargs = args[args.index("--") + 1:]

    # Get the .obj filename
    obj_path = pyargs[0]

    # Put together output path
    if len(pyargs) < 2:
        out_path = os.path.splitext(obj_path)[0] + ".egg"
    else:
        # Get the .<out> filename from cmd line
        out_path = pyargs[1]

    # Put together rotation
    if len(pyargs) < 3:
        rot = None
    else:
        # Get rotation from cmd line
        param_str = pyargs[2]
        rot = [float(s) for s in param_str.split(",")]

    # Run the import
    run(obj_path, out_path, rot=rot)

