#!/usr/bin/env python
"""
Convert .obj models to .obj or .egg (Panda3d format) files
Peter W Battaglia - 03.2012
PWB - 10.2012 - updates
"""
import obj2egg as o2e
import os
import shutil
import sys
import tarfile
from subprocess import call
from subprocess import check_call
import pdb


def blender_convert(ext=".obj"):

    # Model root directory
    model_path = os.path.join(os.environ["HOME"], "work/genthor/models/")

    # Model info scripts
    model_categories_py = "model_categories"
    canonical_angles_py = "canonical_angles"

    # Temporary path in which to extract .obj files before conversion.
    tmp_path = os.path.join(os.environ["HOME"], "tmp", "scrap")

    # Blender script and conversion command.
    blender_path = os.path.join(os.environ["HOME"], "bin", "blender")
    blender_script_name = "obj_Bscript.py"
    blender_command_base = "%s -b -P %s --" % (blender_path,
                                               blender_script_name)

    # Destination directory for .<out> files
    out_root_path = os.path.join(os.environ["HOME"], "tmp", "processed_models")

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

    # Raise exception if the out_root_path is an existing file
    if os.path.isfile(out_root_path):
        raise IOError("File already exists, cannot make: %s" % out_root_path)

    # Create out_root_path directory if necessary
    if not os.path.isdir(out_root_path):
        os.makedirs(out_root_path)

    # Don't rename models 
    outdict = dict(zip(modeldict.keys(), modeldict.keys()))
    # # Rename models as numbered category instances
    # outnames = []
    # for categ, names in model_categories.iteritems():
    #     outnames.extend([(name, categ + str(i))
    #                      for i, name in enumerate(names)])
    #outdict = dict(outnames)

    tgz_paths = []
    
    # Loop over models, doing the conversions
    for modelname, targzname in modeldict.iteritems():
        
        # un-tar, un-gz into a temp directory
        fulltargzname = os.path.join(model_path, targzname)
        allnames = untar(tmp_path, fulltargzname)

        objname = modelname + ".obj"
        # Get target's path
        names = [n for n in allnames if os.path.split(n)[1] == objname]
        # Raise exception if there are not exactly 1
        if len(names) != 1:
            raise ValueError("Cannot find unique object file in tar. Found: %s"
                             % ", ".join(names))
        # Construct obj and out paths
        obj_path = os.path.join(tmp_path, names[0])
        outname = outdict[modelname]
        out_paths = []
        out_paths.append(os.path.join(out_root_path, outname,
                                      outname + ext))
        out_paths.append(os.path.join(os.path.split(out_paths[0])[0], "tex"))
        if ext == ".obj":
            out_paths.append(os.path.join(out_root_path, outname,
                                          outname + ".mtl"))

        # if outname not in ("bloodhound", "MB29826"):
        #     continue
        # else:
        #     #pdb.set_trace()
        #     pass

        # The params are the angles
        params = angledict[modelname]
        
        # Do the conversion from .obj to .<out>
        call_blender(obj_path, out_paths[0], blender_command_base, params)

        # Convert the .<out> to a .tgz
        outtgz_path = os.path.splitext(out_paths[0])[0] + ".tbz2"
        with tarfile.open(outtgz_path, mode="w:bz2") as tf:
            for out_path in out_paths:
                tf.add(out_path, out_path.split(out_root_path + "/")[1])

        # Remove tmp directory
        rm_path = os.path.join(
            tmp_path, obj_path[len(tmp_path.rstrip("/")) + 1:].split("/")[0])
        print "rm -rf %s" % rm_path
        shutil.rmtree(rm_path)
        # Remove .<out> file (eggs can be huge)
        for out_path in out_paths:
            if os.path.isfile(out_path):
                os.remove(out_path)
            elif os.path.isdir(out_path):
                shutil.rmtree(out_path)

        tgz_paths.append(outtgz_path)

    return tgz_paths


def panda_convert(tgz_paths, out_root="", outext=".egg"):
    # Temporary path in which to extract .obj files before conversion.
    tmp_path = os.path.join(os.environ["HOME"], "tmp", "scrap")

    # Loop over models, converting <out>s to bams and deleting the <outs>
    for tgz_path in tgz_paths:
        # un-tar, un-gz into a temp directory
        allnames = untar(tmp_path, tgz_path)

        # Get target's path
        names = [n for n in allnames
                 if os.path.splitext(n)[1] in (".egg", ".obj")]
        # Raise exception if there are not exactly 1
        if len(names) != 1:
            raise ValueError("Cannot find unique object file in tar. Found: %s"
                             % ", ".join(names))
        name = names[0]
        # File extension
        ext = os.path.splitext(name)[1]

        # make the input/output paths
        in_path = os.path.join(tmp_path, name)
        bam_path = os.path.join(out_root, os.path.splitext(name)[0] + ".bam")

        if ext == ".obj":
            # Convert .obj to .bam
            egg_path = os.path.join(out_root, os.path.splitext(name)[0] + ".egg")
            # Convert .obj to .egg
            obj2egg(in_path, egg_path=egg_path)
            if outext == ".bam":
                # Convert .egg to .bam
                egg2bam(egg_path, bam_path=bam_path)
                os.remove(egg_path)
        elif ext == ".egg" and outext == ".bam":
            # Convert .egg to .bam
            egg2bam(in_path, bam_path=bam_path)
        else:
            raise ValueError("unsupported output type: %s" % ext)

        # Remove tmp directory
        rm_path = os.path.join(
            tmp_path, in_path[len(tmp_path.rstrip("/")) + 1:].split("/")[0])
        print "rm -rf %s" % rm_path
        shutil.rmtree(rm_path)


def untar(tmp_path, tarname):
    # Make the tmp_path directory if it doesn't exist already
    if not os.path.isdir(tmp_path):
        os.makedirs(tmp_path)

    # Open the tar
    with tarfile.open(tarname, 'r') as tf:
        # Extract it
        tf.extractall(tmp_path)
        # Get tar.gz's member names
        names = tf.getnames()

    return names


def call_blender(obj_path, out_path, blender_command_base, params):
    # Split into directory and filename
    outdir, outname = os.path.split(out_path)
    
    # Make the outdir directory if it doesn't exist already
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # Put the parameters together into a string
    param_str = ",".join([str(float(x)) for x in params])

    # Assemble the full blender command
    blender_command = "%s %s %s %s" % (blender_command_base, obj_path,
                                       out_path, param_str)

    # Run the blender conversion
    try:
        check_call(blender_command, shell=True)
    except Exception as details:
        print "Tried to call: "
        print blender_command
        print
        print "Failed with exception: %s" % details
        pdb.set_trace()

    # Copy the textures from the .obj path to the .<out> path
    tex_path = os.path.join(os.path.split(out_path)[0], "tex")
    copy_tex(os.path.split(obj_path)[0], tex_path)


def obj2egg(obj_path, egg_path=None):
    ## Make an .egg
    if egg_path is None:
        egg_path = os.path.splitext(obj_path)[0] + '.egg'
    #call("obj2egg -o %s %s" % (egg_path, obj_path), shell=True)
    o2e.main(argv=["", obj_path])
    egg_path0 = egg_path = os.path.splitext(obj_path)[0] + '.egg'
    if egg_path0 != egg_path:
        shutil.move(egg_path0, egg_path)
        pth0 = os.path.split(obj_path)[0]
        pth = os.path.split(egg_path)[0]
        shutil.copytree(os.path.join(pth0, "tex"), pth)


def egg2bam(egg_path, bam_path=None):
    # Make a .bam
    if bam_path is None:
        bam_path = os.path.splitext(egg_path)[0] + '.bam'
    call("egg2bam -o %s %s" % (bam_path, egg_path), shell=True)


def copy_tex(obj_path, tex_path):
    """ Copy texture images from .obj's directory to .egg's directory """

    # Texture image file extensions
    imgexts = (".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".png")
    
    # Tex image files in obj_path
    tex_filenames0 = [name for name in os.listdir(obj_path)
                      if os.path.splitext(name)[1].lower() in imgexts]

    # Make the directory if need be, and error if it is a file already
    if os.path.isfile(tex_path):
        raise IOError("File exists: '%s'")
    elif not os.path.isdir(tex_path):
        os.mkdir(tex_path)

    for name in tex_filenames0:
        #print "%s --> %s" % (os.path.join(obj_path, name), tex_path)
        new_tex_path = os.path.join(tex_path, name.lower())
        shutil.copy2(os.path.join(obj_path, name), new_tex_path)


def main():
    ext = ".obj"
    # Make the egg files from the objs
    tgz_pairs = blender_convert(ext)
    # Make the .egg/.bam files from the eggs
    panda_convert(tgz_pairs, outext=".egg")
    

if __name__ == "__main__":
    main()
