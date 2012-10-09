"""
Blender script for normalizing .obj files and converting them to either .obj or
.egg.
"""
import os
import re
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


def fix_tex_names(mtl_path, imgdirname="tex", f_verify=True):
    """ Make all .mtl image file names lowercase and relative paths,
    so they are compatible with linux and are portable.  Also change
    the actual image file names."""

    # mtl_path = "/home/pbatt/tmp/3dmodels/MB26897/MB26897.mtl"
    # #mtl_path = "/home/pbatt/tmp/3dmodels/MB29698/MB29698.mtl"
    # mtl_path = "/home/pbatt/tmp/iguana/iguana.mtl"

    # Texture image file extensions
    img_exts = (".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".png")

    # .mtl image fields
    mtl_img_fields = ("map_Ka", "map_Kd", "map_bump", "bump", "map_refl")

    # Directory path that the .mtl file is in
    dir_path = os.path.split(mtl_path)[0]

    # Name of this directory
    dir_name = os.path.split(dir_path)[1]

    # Directory for the images
    img_pth = os.path.join(dir_path, imgdirname)

    # visit() function for os.path.walk().  It appends detected image
    # files to a list.
    def visit(imgnames, dir_pth, names):
        imgnames.extend([os.path.join(dir_pth, name) for name in names
                         if os.path.splitext(name)[1].lower() in img_exts])
    # walk down directory tree and get the image files    
    imgpaths0 = []
    for dp, foo, names in os.walk(dir_path):
        visit(imgpaths0, dp, names)
    imgnames = [os.path.split(pth)[1].lower() for pth in imgpaths0]

    # RE pattern
    pat_fields = "(?:" + "|".join(mtl_img_fields) + ") "
    pat_img_exts = "(.+(?:\\" + "|\\".join(img_exts) + "))"
    patstr = "[\s]*" + pat_fields + "((?:.*[/\\\\])?" + pat_img_exts + ")"
    rx = re.compile(patstr, re.IGNORECASE)

    # Initialize storage for the image file names inside the .mtl
    mtlnames = []
    mtllines = []
   
    ## Get the image file names from the .mtl file
    # Open .mtl
    with open(mtl_path, "r") as fid:
        # Iterate over lines
        for line in fid.readlines():
            # Search the line
            m = rx.search(line)
            if m is not None:
                # Pull out the path and image name
                pth = m.group(1)
                Name = m.group(2)
                name = Name.lower()
                # If an image file name is found, store it
                mtlnames.append(name)
                # Edit the line and store
                newline = (line[:m.start(1)] + os.path.join(imgdirname, name)
                           + line[m.end(1):])
                mtllines.append(newline)
            else:
                mtllines.append(line)

    ## Edit .mtl files
    # Open .mtl
    with open(mtl_path, "w") as fid:
        # Iterate over lines
        for line in mtllines:
            # Write the line
            fid.write(line)
                
    # Make unique and sort
    mtlnames = sorted(set(mtlnames))

    if f_verify:
        # Verify that all the mtl images are present
        for mtlname in mtlnames:
            if mtlname not in imgnames:
                raise ValueError("Cannot find .mtl-defined image. "
                                 "mtl: %s. img: %s" % (mtl_path, mtlname))

    # Make the directory if need be, and error if it is a file already
    if os.path.isfile(img_pth):
        raise IOError("File exists: '%s'")
    elif not os.path.isdir(img_pth):
        # Make image directory, if necessary
        os.makedirs(img_pth)
    
    # Move the image files to the new img_pth location
    for imgpath0, imgname in zip(imgpaths0, imgnames):
        imgpath = os.path.join(img_pth, imgname)
        shutil.move(imgpath0, imgpath)
        #print "%s --> %s" % (imgpath0, imgpath)


def import_obj(pth):
    # Fix the .mtl and texture names
    mtl_path = os.path.splitext(pth)[0] + ".mtl"
    fix_tex_names(mtl_path)
    # Import the .obj
    bpy.ops.import_scene.obj(filepath=pth)


def export_obj(pth):
    bpy.ops.export_scene.obj(filepath=pth, use_normals=True,
                             keep_vertex_order=True) 
    
    # Fix the .mtl and texture names
    mtl_path = os.path.splitext(pth)[0] + ".mtl"
    fix_tex_names(mtl_path)

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


def run(obj_path, out_path, rot):
    # Empty the current scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    
    # Import obj into scene
    import_obj(obj_path)

    # Do whatever you need to do within the Blender scene
    transform_model(rot)

    ext = os.path.splitext(out_path)[1]
    if ext == ".egg":
        # Export egg
        export_egg(out_path)
    elif ext == ".obj":
        # Export obj
        if obj_path == out_path:
            # don't clobber
            raise ValueError("I cannot overwrite the obj files: %s" % out_path)
        # copy textures
        shutil.copytree(os.path.join(os.path.split(obj_path)[0], "tex"),
                        os.path.join(os.path.split(out_path)[0], "tex"))
        # export the obj
        export_obj(out_path)
    else:
        raise ValueError("unsupported output type: %s" % ext)

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

    # Put together output path
    if len(pyargs) < 2:
        out_path = os.path.splitext(obj_path)[0] + ".egg"
    else:
        # Get the .<out> filename from cmd line
        out_path = pyargs[1]

    # Put together rotation
    if len(pyargs) < 3:
        rot = [0., 0., 0.]
    else:
        # Get rotation from cmd line
        param_str = pyargs[2]
        rot = [float(s) for s in param_str.split(",")]

    # Run the import
    run(obj_path, out_path, rot)

