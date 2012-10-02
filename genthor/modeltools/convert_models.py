#!/usr/bin/env python

"""
Convert .obj models to .egg files (Panda3d format)
Peter Battaglia - 03.2012

"""

import os
import shutil
import sys
import tarfile
#from math import pi
from subprocess import call
from subprocess import check_call
import pdb


def main():

    # Model root directory
    model_path = os.path.join(os.environ["HOME"], "work/genthor/models/")

    # Model info scripts
    model_categories_py = "model_categories"
    canonical_angles_py = "canonical_angles"

    # Temporary path in which to extract .obj files before conversion.
    tmp_path = os.path.join(os.environ["HOME"], "tmp", "scrap")

    # Blender script and conversion command.
    blender_path = os.path.join(os.environ["HOME"], "bin", "blender")
    blender_script_name = "obj2egg.py"
    blender_command_base = "%s -b -P %s --" % (blender_path,
                                               blender_script_name)

    # Destination directory for .egg files
    egg_root_path = os.path.join(os.environ["HOME"], "tmp", "processed_models")

    # Get the model names from the directory listing
    modeldict = dict([(name[:-7], name) for name in os.listdir(model_path)
                      if name[-7:] == ".tar.gz"])

    # Get the model info that's contained in the scripts
    sys.path.append(model_path)
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

    # Raise exception if the egg_root_path is an existing file
    if os.path.isfile(egg_root_path):
        raise IOError("File already exists, cannot make dir: %s"
                      % egg_root_path)

    # Create egg_root_path directory if necessary
    if not os.path.isdir(egg_root_path):
        os.makedirs(egg_root_path)

    # Don't rename models 
    eggdict = dict(zip(modeldict.keys(), modeldict.keys()))
    # # Rename models as numbered category instances
    # eggnames = []
    # for categ, names in model_categories.iteritems():
    #     eggnames.extend([(name, categ + str(i))
    #                      for i, name in enumerate(names)])
    #eggdict = dict(eggnames)

    name_tgz_pairs = []
    
    # Loop over models, doing the conversions
    for modelname, targzname in modeldict.iteritems():
        
        # un-tar, un-gz into a temp directory
        fulltargzname = os.path.join(model_path, targzname)
        objname = untar(tmp_path, fulltargzname, modelname + ".obj")

        # Construct obj and egg paths
        obj_path = os.path.join(tmp_path, objname)
        eggname = eggdict[modelname]
        egg_path = os.path.join(egg_root_path, eggname, eggname + '.egg')

        # if eggname not in ("bloodhound", "MB29826"):
        #     continue
        # else:
        #     #pdb.set_trace()
        #     pass

        # The params are the angles
        params = angledict[modelname]
        
        # Do the conversion from .obj to .egg
        convert(obj_path, egg_path, blender_command_base, params)

        # Convert the .egg to a .tgz
        outtgz_path = os.path.splitext(egg_path)[0] + ".tbz2"
        with tarfile.open(outtgz_path, mode="w:bz2") as tf:
            tf.add(egg_path, egg_path.split(egg_root_path + "/")[1])
            tex_path = os.path.join(os.path.split(egg_path)[0], "tex")
            tf.add(tex_path, tex_path.split(egg_root_path + "/")[1])

        # Remove tmp directory
        rm_path = os.path.join(
            tmp_path, obj_path[len(tmp_path.rstrip("/")) + 1:].split("/")[0])
        print "rm -rf %s" % rm_path
        shutil.rmtree(rm_path)

        name_tgz_pairs.append((eggname, outtgz_path))

        # # Convert .egg to .bam
        # egg2bam(egg_path)
        
        # Remove .egg file (because they're huge)
        os.remove(egg_path)


    # Loop over models, converting eggs to bams and deleting the eggs
    for eggname, outtgz_path in name_tgz_pairs:
        
        # un-tar, un-gz into a temp directory
        egg_path0 = untar(tmp_path, outtgz_path, eggname + ".egg")
        # make the input/output paths
        egg_path = os.path.join(tmp_path, egg_path0)
        bam_path = os.path.join(os.path.splitext(outtgz_path)[0], ".bam")
        
        # Convert .egg to .bam
        egg2bam(egg_path, bam_path=bam_path)

        # Remove tmp directory
        rm_path = os.path.join(
            tmp_path, obj_path[len(tmp_path.rstrip("/")) + 1:].split("/")[0])
        print "rm -rf %s" % rm_path
        shutil.rmtree(rm_path)




# def untargz_obj(tmp_path, targzname, modelname):
#     # Make the tmp_path directory if it doesn't exist already
#     if not os.path.isdir(tmp_path):
#         os.makedirs(tmp_path)

#     # .obj filename
#     objname = modelname + ".obj"

#     # Open the tar.gz
#     with tarfile.open(targzname, 'r') as tf:
#         # Extract it
#         tf.extractall(tmp_path)
#         # Get tar.gz's member names
#         tarnames = tf.getnames()

#     # Get obj_path
#     obj_path = [pth for pth in tarnames
#                 if os.path.split(pth)[1] == objname]
#     # Raise exception if there're more than 1
#     if len(obj_path) != 1:
#         raise ValueError("Cannot find unique .obj file in tar.gz. Found: %s"
#                          % ", ".join(obj_path))
#     else:
#         obj_path = obj_path[0]

#     return obj_path


def untar(tmp_path, tarname, objname=""):
    # Make the tmp_path directory if it doesn't exist already
    if not os.path.isdir(tmp_path):
        os.makedirs(tmp_path)

    # Open the tar
    with tarfile.open(tarname, 'r') as tf:
        # Extract it
        tf.extractall(tmp_path)
        # Get tar.gz's member names
        tarnames = tf.getnames()

    # Get target's path
    names = [name for name in tarnames
             if os.path.split(name)[1].endswith(objname)]
    # Raise exception if there're more than 1
    if len(names) != 1:
        raise ValueError("Cannot find unique object file in tar. Found: %s"
                         % ", ".join(names))
    else:
        name = names[0]

    return name


def convert(obj_path, egg_path, blender_command_base, params):
    # Split into directory and filename
    eggdir, eggname = os.path.split(egg_path)
    
    # Make the eggdir directory if it doesn't exist already
    if not os.path.isdir(eggdir):
        os.makedirs(eggdir)

    # Put the parameters together into a string
    param_str = ",".join([str(float(x)) for x in params])

    # Assemble the full blender command
    blender_command = "%s %s %s %s" % (blender_command_base, obj_path,
                                       egg_path, param_str)

    # Run the blender conversion
    try:
        check_call(blender_command, shell=True)
    except Exception as details:
        print "Tried to call: "
        print blender_command
        print
        print "Failed with exception: %s" % details
        pdb.set_trace()

    # Copy the textures from the .obj path to the .egg path
    tex_path = os.path.join(os.path.split(egg_path)[0], "tex")
    copy_tex(os.path.split(obj_path)[0], tex_path)


def egg2bam(egg_path, bam_path=None):
    # Make a .bam copy
    if bam_path is None:
        bam_path = os.path.splitext(egg_path)[0] + '.bam'
    call("egg2bam -o %s %s" % (bam_path, egg_path), shell=True)


def copy_tex(obj_path, tex_path):
    """ Copy texture images from .obj's directory to .egg's directory """

    # Texture image file extensions
    tex_imgexts = (".jpg", ".tif", ".bmp", ".gif", ".png")
    
    # Tex image files in obj_path
    tex_filenames0 = [name for name in os.listdir(obj_path)
                      if os.path.splitext(name)[1].lower() in tex_imgexts]

    # Make the directory if need be, and error if it is a file already
    if os.path.isfile(tex_path):
        raise IOError("File exists: '%s'")
    elif not os.path.isdir(tex_path):
        os.mkdir(tex_path)

    for name in tex_filenames0:
        #print "%s --> %s" % (os.path.join(obj_path, name), tex_path)
        shutil.copy2(os.path.join(obj_path, name), tex_path)
    

if __name__ == "__main__":

    main()
