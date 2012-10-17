#!/usr/bin/env python
"""
Convert .obj models to .obj, .egg/.bam (Panda3d format) files
Peter W Battaglia - 03.2012
PWB - 10.2012 - updates
"""
import genthor as gt
import genthor.modeltools.tools as mt
import numpy as np
#import obj2egg as o2e
import os
import shutil
from subprocess import call
from subprocess import check_call
import sys
import tarfile
import pdb


# ## OLD OBJ2EGG THAT USES obj2egg.py, which is a garbage script
# def obj2egg(obj_pth, egg_pth=None, f_force_tex=True):
#     ## Make an .egg
#     if egg_pth is None:
#         egg_pth = os.path.splitext(obj_pth)[0] + '.egg'
#     # Call obj2egg.py script to convert
#     o2e.main(argv=["", obj_pth, "-b -t"])
#     egg_pth0 = os.path.splitext(obj_pth)[0] + '.egg'
#     if egg_pth0 != egg_pth:
#         shutil.move(egg_pth0, egg_pth)
#         pth0 = os.path.split(obj_pth)[0]
#         pth = os.path.split(egg_pth)[0]
#         tex_pth = os.path.join(pth, "tex")
#         if os.path.isdir(tex_pth):
#             if f_force_tex:
#                 shutil.rmtree(tex_pth)
#             else:
#                 raise IOError("Directory exists: %s" % tex_pth)
#         shutil.copytree(os.path.join(pth0, "tex"), tex_pth)


def obj2egg(obj_pth, egg_pth=None, f_blender=True): #, f_force_tex=True):
    """ Convert an .obj file to an .egg using Blender."""
    if not f_blender:
        raise ValueError("f_blender=False is unsupported")
    ## Make an .egg
    if egg_pth is None:
        egg_pth = os.path.splitext(obj_pth)[0] + '.egg'
    # Blender script and conversion command.
    blender_pth = os.path.join(os.environ["HOME"], "bin", "blender")
    blender_script_name = os.path.join(gt.GENTHOR_PATH, "obj_Bscript.py")
    blender_command_base = "%s -b -P %s --" % (blender_pth, blender_script_name)

    ## Do the conversion from .obj to .egg using Blender
    # Run the blender script
    call_blender(obj_pth, egg_pth, blender_command_base)

    # egg_pth0 = os.path.splitext(obj_pth)[0] + '.egg'
    # if egg_pth0 != egg_pth:
    #     shutil.move(egg_pth0, egg_pth)
    #     pth0 = os.path.split(obj_pth)[0]
    #     pth = os.path.split(egg_pth)[0]
    #     tex_pth = os.path.join(pth, "tex")
    #     if os.path.isdir(tex_pth):
    #         if f_force_tex:
    #             shutil.rmtree(tex_pth)
    #         else:
    #             raise IOError("Directory exists: %s" % tex_pth)
    #     shutil.copytree(os.path.join(pth0, "tex"), tex_pth)


def egg2bam(egg_pth, bam_pth=None):
    # Make a .bam
    if bam_pth is None:
        bam_pth = os.path.splitext(egg_pth)[0] + '.bam'
    call("egg2bam -o %s %s" % (bam_pth, egg_pth), shell=True)


def get_modeldata(model_pth):

    # Model info scripts
    model_categories_py = "model_categories"
    canonical_angles_py = "canonical_angles"

    # Get the model names from the directory listing
    modeldict = dict([(name[:-7], os.path.join(model_pth, name))
                      for name in os.listdir(model_pth)
                      if name[-7:] == ".tar.gz"])

    # Get the model info that's contained in the scripts
    sys.path.append(model_pth)
    model_categories = __import__(model_categories_py).MODEL_CATEGORIES
    canonical_angles = __import__(canonical_angles_py).ANGLES

    # Assemble category info in dict with {modelname: category, ...}
    categories = []
    for categ, names in model_categories.iteritems():
        categories.extend([(name, categ) for name in names])
    categorydict = dict(categories)

    # Assemble angle info in dict with {modelname: angle, ...}
    angledict = dict([(entry[0], entry[1:]) for entry in canonical_angles])

    # Check that model_categories and canonical_angles has info on the models
    modelnames = set(modeldict.keys())
    # model_categories
    names = set(categorydict.keys())
    if not modelnames.issubset(names):
        raise ValueError("%s doesn't have info for: %s" % (
            model_categories_py, ",".join(modelnames.difference(names))))
    # canonical_angles
    names = set(angledict.keys())
    if not modelnames.issubset(names):
        raise ValueError("%s does not have info for: %s" % (
            canonical_angles_py, ", ".join(modelnames.difference(names))))

    # Don't rename models 
    outdict = dict(zip(modeldict.keys(), modeldict.keys()))
    # # Rename models as numbered category instances
    # outnames = []
    # for categ, names in model_categories.iteritems():
    #     outnames.extend([(name, categ + str(i))
    #                      for i, name in enumerate(names)])
    #outdict = dict(outnames)

    return modeldict, outdict, angledict


def build_objs(out_root, modeldict, outdict, angledict=None,
               f_tgz=True, f_force=False):

    # Temporary path in which to extract .obj files before conversion.
    tmp_root = os.path.join(os.environ["HOME"], "tmp", "scrap")

    # Raise exception if the out_root is an existing file
    if os.path.isfile(out_root):
        raise IOError("File already exists, cannot make: %s" % out_root)

    # Create out_root directory if necessary
    if not os.path.isdir(out_root):
        os.makedirs(out_root)

    imgdirname = "tex"
    out_pths = []
    # Loop over models, doing the conversions
    for modelname, targzname in modeldict.iteritems():
        # Set up file names
        objname = modelname + ".obj"
        outname = outdict[modelname]
        # Make new paths
        new_obj_pth = os.path.join(out_root, outname, outname + ".obj")
        new_tex_pth = os.path.join(out_root, outname, imgdirname)
        new_mtl_pth = os.path.join(out_root, outname, outname + ".mtl")
        if f_tgz:
            out_pth = os.path.join(out_root, outname, outname + ".tgz")
        else:
            out_pth = new_obj_pth
        out_pths.append(out_pth)
        if not f_force and os.path.isfile(out_pth):
            continue

        print "Building: %s" % objname

        # un-tar, un-gz into a temp directory
        tmp_tar_pth = os.path.join(tmp_root, "tartmp")
        allnames = mt.untar(targzname, tmp_tar_pth)

        # Get target's path
        names = [n for n in allnames if os.path.split(n)[1] == objname]
        # Raise exception if there are not exactly 1
        if len(names) != 1:
            raise ValueError("Cannot find unique object file in tar. Found: %s"
                             % ", ".join(names))
        # Make obj, mtl, and tex paths
        obj_pth = os.path.join(tmp_tar_pth, names[0])
        # Fix the .mtl and texture names and move them
        mtl_pth = os.path.splitext(obj_pth)[0] + ".mtl"
        mt.fix_tex_names(mtl_pth, imgdirname=imgdirname)
        tex_pth = os.path.join(os.path.dirname(obj_pth), imgdirname)

        ## Normalize the .obj coordinates
        # The params are the angles
        if angledict is not None:
            params = angledict[modelname]
            # Transform vertices
            rot = (params[0] + np.pi / 2., params[1] + np.pi / 2., params[2])
            T0 = mt.build_rot(rot)
            # Normalize the obj and move it 
            mt.transform_obj(obj_pth, new_obj_pth, T0=T0)
        else:
            # Normalize the obj and move it 
            mt.transform_obj(obj_pth, new_obj_pth)
        # Copy .mtl file over
        shutil.copy2(mtl_pth, new_mtl_pth)
        # Copy the textures over
        mt.copy_tex(tex_pth, new_tex_pth)

        if f_tgz:
            # Convert the .obj/.mtl/tex to a .tgz
            with tarfile.open(out_pth, mode="w:gz") as tf:
                for out_pth in (new_obj_pth, new_mtl_pth, new_tex_pth):
                    # Add to zip
                    tf.add(out_pth, out_pth.split(out_root + "/")[1])
                    # Remove the files
                    if os.path.isfile(out_pth):
                        os.remove(out_pth)
                    elif os.path.isdir(out_pth):
                        shutil.rmtree(out_pth)

        # Remove tmp directory
        print "rm -rf %s" % tmp_root
        shutil.rmtree(tmp_root)

    out_pths.sort()
    return out_pths
    

def convert(inout_pths, ext=".egg", f_blender=True, f_force=False):
    if ext not in (".egg", ".bam"):
        raise ValueError("Unsupported output type: %s" % ext)
    # Temporary path in which to extract .obj files before conversion.
    tmp_root = os.path.join(os.environ["HOME"], "tmp", "scrap")

    # Loop over models, converting
    for in_pth, out_pth in inout_pths.iteritems():
        if not f_force and os.path.isfile(out_pth):
            continue
        if not os.path.isfile(in_pth):
            raise IOError("File does not exist: %s" % in_pth)
         
        # Determine file name and extension
        name, ext0 = gt.splitext2(os.path.basename(in_pth))
        if ext0 in (".tgz", ".tar.gz", ".tbz2", ".tar.bz2"):
            # un-tar, un-gz into a temp directory
            tmp_tar_pth = os.path.join(tmp_root, "tartmp")
            allnames = mt.untar(in_pth, tmp_tar_pth)

            # Get target's path
            names = [n for n in allnames
                     if os.path.splitext(n)[1] in (".egg", ".obj")]
            # Raise exception if there are not exactly 1
            if len(names) != 1:
                raise ValueError("Can't find unique file in tar. Found: %s"
                                 % ", ".join(names))
            in_pth = os.path.join(tmp_tar_pth, names[0])
            ext0 = os.path.splitext(names[0])[1]
        elif ext0 not in (".egg", ".obj"):
            raise ValueError("Can't handle extension: %s" % in_pth)

        # Make the output paths. If the specified path has an
        # extension, that takes precedence over the argument 'ext'.
        ext1 = os.path.splitext(out_pth)[1]
        if ext1 == "":
            # out_pth is a directory, make it a filename
            out_pth = os.path.join(out_pth, name + ext)
            ext1 = ext
        elif ext1 not in (".egg", ".bam"):
            raise ValueError("Unsupported output type: %s" % ext1)

        # Make output directory (if necessary)
        if not os.path.isdir(os.path.split(out_pth)[0]):
            os.makedirs(os.path.split(out_pth)[0])

        # Do the conversion, depending on ext1 type
        if ext0 == ".obj":
            if ext1 == ".bam":
                # Two-step conversion, first to egg, then to bam
                egg_pth = os.path.join(tmp_root, "eggtmp", "tmp.egg")
            else:
                # One-step conversion, to egg.
                egg_pth = out_pth
            # Convert .obj to .egg
            if not os.path.isfile(egg_pth):
                obj2egg(in_pth, egg_pth=egg_pth, f_blender=f_blender)
            if ext1 == ".bam":
                # Convert .egg to .bam
                egg2bam(egg_pth, bam_pth=out_pth)
        elif ext0 == ".egg" and ext1 == ".bam":
            # Convert .egg to .bam
            egg2bam(in_pth, bam_pth=out_pth)
        else:
            raise ValueError("unsupported output type: %s" % ext)

        # Remove all tmp directories
        print "rm -rf %s" % tmp_root
        shutil.rmtree(tmp_root)


def autogen_egg(model_pth):
    # modelpth is now a valid file, check its extension and create an
    # .egg if it is not a panda file
    name, ext = gt.splitext2(os.path.basename(model_pth))
    if ext not in mt.panda_exts:
        # modelpth is not a panda3d extension
        # The .egg's path
        egg_pth = os.path.join(gt.EGG_PATH, name, name + ".egg")

        if not os.path.isfile(egg_pth):
            # The .egg doesn't exist, so convert the input file
            inout_pth = {model_pth: egg_pth}
            convert(inout_pth, ext=".egg")
    else:
        egg_pth = model_pth
    return egg_pth


def call_blender(obj_pth, out_pth, blender_command_base, params=None):
    # Split into directory and filename
    outdir, outname = os.path.split(out_pth)
    
    # Make the outdir directory if it doesn't exist already
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if params is not None:
        # Put the parameters together into a string
        param_str = ",".join([str(float(x)) for x in params])
        # Assemble the full blender command
        blender_command = "%s %s %s %s" % (blender_command_base, obj_pth,
                                           out_pth, param_str)
    else:
        # Assemble the full blender command
        blender_command = "%s %s %s" % (blender_command_base, obj_pth, out_pth)
    
    # Run the blender conversion
    try:
        check_call(blender_command, shell=True)
    except Exception as details:
        print "Tried to call: "
        print blender_command
        print
        print "Failed with exception: %s" % details
        pdb.set_trace()


def main(f_egg=True):
    # Model root directory
    model_pth = "/home/pbatt/work/genthor/raw_models"
    # Destination directory for .<out> files
    obj_root = gt.OBJ_PATH
    # Get the necessary data for the models (names, angles, etc)
    modeldict, outdict, angledict = get_modeldata(model_pth)
    # Build the .obj database
    print "Building .obj database..."
    obj_pths = build_objs(obj_root, modeldict, outdict,
                          angledict=angledict, f_tgz=True, f_force=False)
    print "Finishes building .obj database."
    if f_egg:
        ## Convert to .egg
        # Destination directory for .egg files
        egg_root = gt.EGG_PATH
        # Create the in-out paths dict
        egg_pths = [os.path.join(egg_root, gt.splitext2(
            os.path.basename(obj_pth))[0]) for obj_pth in obj_pths]
        inout_pths = dict(zip(obj_pths, egg_pths))
        # Do the conversion
        print "Building .egg database from .obj database..."
        convert(inout_pths, ext=".egg")
        print "Finished building .egg database."


class FormatError(Exception):
    def __init__(self, inpth):
        self.msg = 'Input path %s fails to meet check_format criteria' % inpth
    

if __name__ == "__main__":
    main()


## TODO
#
# Make cmd line interface better using argparse
# 

