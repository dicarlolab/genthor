#!/usr/bin/env python

"""
Convert .obj models to .egg files (Panda3d format)
Peter Battaglia - 03.2012

"""

import os
import shutil
import sys
import tarfile
from subprocess import call
from subprocess import check_call
import pdb


def main():

    # Model root directory
    model_path = os.path.join(os.environ["HOME"], "Dropbox/genthor/models/")

    # Model info scripts
    model_categories_py = "model_categories"
    canonical_angles_py = "canonical_angles"

    # Temporary path in which to extract .obj files before conversion.
    tmp_path = os.path.join(os.environ["HOME"], "tmp", "scrap")

    # Blender script and conversion command.
    blender_path = os.path.join(os.environ["HOME"], "bin", "blender")
    blender_script_name = "obj2egg.py"
    blender_command_base = "%s -b -P %s --" % (blender_path, blender_script_name)

    # Destination directory for .egg files
    egg_root_path = os.path.join(os.environ["HOME"], "tmp", "egg_models")

    # Get the model names from the directory listing
    modeldict = dict([(name[:-7], name) for name in os.listdir(model_path)
                       if name[-7:] == ".tar.gz"])

    # Get the model info that's contained in the scripts
    sys.path.append(model_path)
    try:
        model_categories = __import__(model_categories_py).MODEL_CATEGORIES
        canonical_angles = __import__(canonical_angles_py).ANGLES
    except ImportError:
        raise

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
        raise IOError("File already exists, cannot make dir: %s" % egg_root_path)

    # Create egg_root_path directory if necessary
    if not os.path.isdir(egg_root_path):
        os.makedirs(egg_root_path)

    # Don't rename models 
    eggdict = modeldict
    # Rename models as numbered category instances
    eggnames = []
    for categ, names in model_categories.iteritems():
        eggnames.extend([(name, categ + str(i)) for i, name in enumerate(names)])
    eggdict = dict(eggnames)
   
    # Loop over models, doing the conversions
    for modelname, targzname in modeldict.iteritems():
        # un-tar, un-gz into a temp directory
        fulltargzname = os.path.join(model_path, targzname)
        objname = untargz(tmp_path, fulltargzname, modelname)

        # Construct obj and egg paths
        obj_path = os.path.join(tmp_path, objname)
        eggname = eggdict[modelname]
        egg_path = os.path.join(egg_root_path, eggname, eggname + '.egg')

        # The params are the angles
        params = angledict[modelname]
        
        # Do the conversion from .obj to .egg
        convert(obj_path, egg_path, blender_command_base, params)

        # Remove all contents of tmp_path
        print
        print "rm -rf %s" % tmp_path
        shutil.rmtree(tmp_path)


def untargz(tmp_path, targzname, modelname):
    # Make the tmp_path directory if it doesn't exist already
    if not os.path.isdir(tmp_path):
        os.makedirs(tmp_path)

    # .obj filename
    objname = modelname + ".obj"

    # Open the .tar.gz
    with tarfile.open(targzname, 'r') as tf:
        # Extract it
        tf.extractall(tmp_path)
        # Get .obj file path
        tf.getmembers()
        pdb.set_trace()

    # # Get any files with extension .obj
    # untargzed_path = os.path.join(tmp_path, "3dmodels", modelname)
    # objname = [name for name in os.listdir(untargzed_path)
    #            if os.path.splitext(name)[1] == ".obj"]
    
    # # If there's more than 1 .obj file, pick those that startwith modelname
    # if len(objname) > 1:
    #     objname = [name for name in objname
    #                if name.startswith(modelname)]
    # # If objname still has more than 1 file, error
    # if len(objname) > 1:
    #     raise IOError("Cannot determine .obj filename. Confused between: %s"
    #                   % ", ".join(objname))

    # Assemble full obj path
    obj_path = os.path.join(untargzed_path, objname[0])

    return obj_path


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
        print "Tried to call:"
        print blender_command
        print
        print "Failed with exception: %s" % details
        pdb.set_trace()


    # yabee seems not to want to copy the textures -- I could do it by hand
    # (ie. grab the JPG files and shutil.copy()/.copy2() them ...


    # Make .bam copy as well
    bam_path = os.path.splitext(egg_path)[0] + '.bam'
    call("egg2bam -o %s %s" % (bam_path, egg_path), shell=True)

    


if __name__ == "__main__":

    main()
