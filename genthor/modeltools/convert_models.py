#!/usr/bin/env python
"""
Convert .obj models to .obj, .egg/.bam (Panda3d format) files
Peter W Battaglia - 03.2012
PWB - 10.2012 - updates
"""
import genthor as gt
import genthor.modeltools.tools as mt
import obj2egg as o2e
import os
import shutil
from subprocess import call
from subprocess import check_call
import sys
import tarfile
import pdb


def blender_convert(model_pth, out_root, ext=".obj", f_force=False):

    # Model info scripts
    model_categories_py = "model_categories"
    canonical_angles_py = "canonical_angles"

    # Temporary path in which to extract .obj files before conversion.
    tmp_root = os.path.join(os.environ["HOME"], "tmp", "scrap")

    # Blender script and conversion command.
    blender_pth = os.path.join(os.environ["HOME"], "bin", "blender")
    blender_script_name = "obj_Bscript.py"
    blender_command_base = "%s -b -P %s --" % (blender_pth,
                                               blender_script_name)

    # Get the model names from the directory listing
    modeldict = dict([(name[:-7], name) for name in os.listdir(model_pth)
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

    # Raise exception if the out_root is an existing file
    if os.path.isfile(out_root):
        raise IOError("File already exists, cannot make: %s" % out_root)

    # Create out_root directory if necessary
    if not os.path.isdir(out_root):
        os.makedirs(out_root)

    # Don't rename models 
    outdict = dict(zip(modeldict.keys(), modeldict.keys()))
    # # Rename models as numbered category instances
    # outnames = []
    # for categ, names in model_categories.iteritems():
    #     outnames.extend([(name, categ + str(i))
    #                      for i, name in enumerate(names)])
    #outdict = dict(outnames)

    tgz_pths = []
    
    # Loop over models, doing the conversions
    for modelname, targzname in modeldict.iteritems():
        # Set up file names
        objname = modelname + ".obj"
        outname = outdict[modelname]
        outtgz_pth = os.path.join(out_root, outname, outname + ".tgz")
        tgz_pths.append(outtgz_pth)
        if not f_force and os.path.isfile(outtgz_pth):
            continue
        
        out_pths = []
        out_pths.append(os.path.join(out_root, outname, outname + ext))
        out_pths.append(os.path.join(out_root, outname, "tex"))
        if ext == ".obj":
            out_pths.append(os.path.join(out_root, outname, outname + ".mtl"))
        
        # un-tar, un-gz into a temp directory
        fulltargzname = os.path.join(model_pth, targzname)
        tmp_tar_pth = os.path.join(tmp_root, "tartmp")
        allnames = mt.untar(fulltargzname, tmp_tar_pth)

        # Get target's path
        names = [n for n in allnames if os.path.split(n)[1] == objname]
        # Raise exception if there are not exactly 1
        if len(names) != 1:
            raise ValueError("Cannot find unique object file in tar. Found: %s"
                             % ", ".join(names))
        # Construct obj and out paths
        obj_pth = os.path.join(tmp_tar_pth, names[0])

        # The params are the angles
        params = angledict[modelname]

        ## Do the conversion from .obj to .<out>
        # Fix the .mtl and texture names
        mtl_path = os.path.splitext(obj_pth)[0] + ".mtl"
        mt.fix_tex_names(mtl_path)
        # Run the blender script
        call_blender(obj_pth, out_pths[0], blender_command_base, params)
        # Fix the .mtl and texture names
        mtl_path = os.path.splitext(out_pths[0])[0] + ".mtl"
        mt.fix_tex_names(mtl_path)
        # Copy the textures from the .obj path to the .<out> path
        tex_pth = os.path.join(os.path.split(out_pths[0])[0], "tex")
        copy_tex(os.path.split(obj_pth)[0], tex_pth)


        # Convert the .<out> to a .tgz
        with tarfile.open(outtgz_pth, mode="w:gz") as tf:
            for out_pth in out_pths:
                tf.add(out_pth, out_pth.split(out_root + "/")[1])

        # Remove tmp directory
        print "rm -rf %s" % tmp_root
        shutil.rmtree(tmp_root)

        # Remove .<out> file (eggs can be huge)
        for out_pth in out_pths:
            if os.path.isfile(out_pth):
                os.remove(out_pth)
            elif os.path.isdir(out_pth):
                shutil.rmtree(out_pth)

    tgz_pths.sort()

    return tgz_pths


def panda_convert(inout_pths, ext=".egg"):
    # Temporary path in which to extract .obj files before conversion.
    tmp_root = os.path.join(os.environ["HOME"], "tmp", "scrap")

    if ext not in (".egg", ".bam"):
        raise ValueError("Unsupported output type: %s" % ext)

    # Loop over models, converting <out>s to bams and deleting the <outs>
    for in_pth, out_pth in inout_pths.iteritems():
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
            obj2egg(in_pth, egg_pth=egg_pth)
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


def autogen_egg(modelpth):
    # modelpth is now a valid file, check its extension and create an
    # .egg if it is not a panda file
    name, ext = gt.splitext2(os.path.basename(modelpth))
    if ext not in mt.panda_exts:
        # modelpth is not a panda3d extension
        # The .egg's path
        pandapth = os.path.join(gt.EGG_PATH, name, name + ".egg")

        if not os.path.isfile(pandapth):
            # The .egg doesn't exist, so convert the input file
            inout_pth = {modelpth: pandapth}
            panda_convert(inout_pth, ext=".egg")
    else:
        pandapth = modelpth
    return pandapth


def call_blender(obj_pth, out_pth, blender_command_base, params):
    # Split into directory and filename
    outdir, outname = os.path.split(out_pth)
    
    # Make the outdir directory if it doesn't exist already
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # Put the parameters together into a string
    param_str = ",".join([str(float(x)) for x in params])

    # Assemble the full blender command
    blender_command = "%s %s %s %s" % (blender_command_base, obj_pth,
                                       out_pth, param_str)

    # Run the blender conversion
    try:
        check_call(blender_command, shell=True)
    except Exception as details:
        print "Tried to call: "
        print blender_command
        print
        print "Failed with exception: %s" % details
        pdb.set_trace()


def obj2egg(obj_pth, egg_pth=None, f_force_tex=True):
    ## Make an .egg
    if egg_pth is None:
        egg_pth = os.path.splitext(obj_pth)[0] + '.egg'
    # Call obj2egg.py script to convert
    o2e.main(argv=["", obj_pth])
    egg_pth0 = os.path.splitext(obj_pth)[0] + '.egg'
    if egg_pth0 != egg_pth:
        shutil.move(egg_pth0, egg_pth)
        pth0 = os.path.split(obj_pth)[0]
        pth = os.path.split(egg_pth)[0]
        tex_pth = os.path.join(pth, "tex")
        if os.path.isdir(tex_pth):
            if f_force_tex:
                shutil.rmtree(tex_pth)
            else:
                raise IOError("Directory exists: %s" % tex_pth)
        shutil.copytree(os.path.join(pth0, "tex"), tex_pth)


def egg2bam(egg_pth, bam_pth=None):
    # Make a .bam
    if bam_pth is None:
        bam_pth = os.path.splitext(egg_pth)[0] + '.bam'
    call("egg2bam -o %s %s" % (bam_pth, egg_pth), shell=True)


def copy_tex(obj_pth, tex_pth):
    """ Copy texture images from .obj's directory to .egg's directory """

    # Tex image files in obj_pth
    tex_filenames0 = [name for name in os.listdir(obj_pth)
                      if os.path.splitext(name)[1].lower() in mt.img_exts]

    # Make the directory if need be, and error if it is a file already
    if os.path.isfile(tex_pth):
        raise IOError("File exists: '%s'")
    elif not os.path.isdir(tex_pth):
        os.makedirs(tex_pth)

    for name in tex_filenames0:
        new_tex_pth = os.path.join(tex_pth, name)
        shutil.copy2(os.path.join(obj_pth, name), new_tex_pth)


def main(f_panda=True):
    # Model root directory
    model_pth = "/home/pbatt/work/genthor/raw_models"
    # Destination directory for .<out> files
    out_root = gt.OBJ_PATH
    # Make the .obj/.egg files from original .obj files
    tgz_pths = blender_convert(model_pth, out_root, ext=".obj")

    if f_panda:
        # Destination directory for .egg files
        out_root = gt.EGG_PATH
        # Make the .egg/.bam files from the .obj/.egg files
        # inout_pths = {}
        # inout_pths[tgz_pth] = os.path.join(out_root, os.path.splitext(
        #     os.path.basename(tgz_pth))[1])
        out_pths = [os.path.join(out_root, gt.splitext2(
            os.path.basename(tgz_pth))[0]) for tgz_pth in tgz_pths]
        inout_pths = dict(zip(tgz_pths, out_pths))

        panda_convert(inout_pths, ext=".egg")


class FormatError(Exception):
    def __init__(self, inpth):
        self.msg = 'Input path %s fails to meet check_format criteria' % inpth
    

if __name__ == "__main__":
    main()


## TODO
#
# Make cmd line interface better using argparse
# 

