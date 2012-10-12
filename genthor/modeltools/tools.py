""" General tools for manipulating models."""
import genthor as gt
import os
import re
import shutil
import tarfile

# Panda extensions
panda_exts = (".bam", ".egg")
# Model extensions
model_exts = (".obj",) + panda_exts
# Zip extensions
zip_exts = (".tgz", ".tar.gz", ".tbz2", ".tar.bz2")
# Texture image file extensions
img_exts = (".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".png")
# .mtl image fields
mtl_img_fields = ("map_ka", "map_kd", "map_ks", "map_ks", "map_d", "disp",
                  "decal", "map_bump", "bump", "map_refl")


def resolve_bg_path(bgpth0):
    """ Finds a valid background path."""
    if not os.path.isfile(bgpth0):
        bgpth = os.path.join(gt.BACKGROUND_PATH, os.path.basename(bgpth0))
        if not os.path.isfile(bgpth):
            raise IOError("Not a valid background file: %s" % bgpth)
    else:
        bgpth = bgpth0
    return bgpth
    

def resolve_model_path(modelpth0):
    """ Finds a valid model path. It will convert an .obj model to an
    .egg if necessary."""
    valid_exts = model_exts + zip_exts
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

    return modelpth


def parse_dir_imgs(root_pth):
    """ Search through pth and all sub-directories for image files and
    return a list of their names."""
    def visit(imgpths, pth, names):
        # Appends detected image filenames to a list.
        imgpths.extend([os.path.join(pth, name) for name in names
                        if os.path.splitext(name)[1].lower() in img_exts])
    # Walk down directory tree and get the image file paths
    imgpaths = []
    for dp, foo, names in os.walk(root_pth):
        visit(imgpaths, dp, names)
    # Make lowercased list of imagefilenames
    imgnames = [os.path.split(pth)[1].lower() for pth in imgpaths]
    return imgnames, imgpaths


def parse_mtl_imgs(mtl_pth, f_edit=False, imgdirname="tex"):
    """ Search through the mtl_pth for all image file names and return
    as a list.  f_edit will substitute fixed img names into the .mtl
    file."""
    # RE pattern
    pat_fields = "(?:" + "|".join(mtl_img_fields) + ") "
    pat_img_exts = "(.+(?:\\" + "|\\".join(img_exts) + "))"
    patstr = "[\s]*" + pat_fields + "((?:.*[/\\\\])?" + pat_img_exts + ")"
    rx = re.compile(patstr, re.IGNORECASE)
    ## Get the image file names from the .mtl file
    # Open .mtl
    with open(mtl_pth, "r") as fid:
        filestr = fid.read()
    if f_edit:
        def repl(m):
            # Get the old file name
            name = m.group(2).lower()
            # Store name
            mtlnames.append(name)
            # Pull out the path and image name
            newname = os.path.join(imgdirname, name)
            # First and last points in match
            i = m.start(1) - m.start()
            j = m.end(1) - m.start()
            match = m.group()
            # Make a substitute for the match
            newmatch = match[:i] + newname + match[j:]
            return newmatch
        # Initialize storage for the image file names from inside the
        # .mtl, which will be appended to by the repl function
        mtlnames = []
        # Search for matches and substitute in fixed path
        newfilestr = rx.subn(repl, filestr)[0]
        # Edit and save new .mtl
        with open(mtl_pth, "w") as fid:
            fid.write(newfilestr)
    else:
        # The .mtl's image filenames
        mtlnames = [m.group(2).lower() for m in rx.finditer(filestr)]
    return mtlnames


def fix_tex_names(mtl_pth, imgdirname="tex", f_verify=True):
    """ Make all .mtl image file names lowercase and relative paths,
    so they are compatible with linux and are portable.  Also change
    the actual image file names."""

    if not os.path.isabs(mtl_pth):
        mtl_pth = os.path.join(os.getcwd(), mtl_pth)
    # Directory path that the .mtl file is in
    dir_pth = os.path.split(mtl_pth)[0]
    # Directory for the images to go
    img_pth = os.path.join(dir_pth, imgdirname)

    # Parse the .mtl file for the image names
    mtlnames = parse_mtl_imgs(mtl_pth, f_edit=True, imgdirname=imgdirname)
    # Get image file names
    imgnames, imgpaths0 = parse_dir_imgs(dir_pth)
    # Check that all .mtl img names are present
    if f_verify and not set(mtlnames) <= set(imgnames):
        missingnames = ", ".join(set(imgnames) - set(mtlnames))
        raise ValueError("Cannot find .mtl-defined images. "
                         "mtl: %s. imgs: %s" % (mtl_pth, missingnames))

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


def check_format(pth, imgdirname="tex"):
    """ Checks format of a model directory to make sure everything is
    formatted per the genthor standard.

    Exit codes:
    0 : Pass
    1 : Couldn't find model
    2 : Couldn't find required texture images
    """
    def returnfunc():
        # Remove tmp path
        if os.path.isdir(tmp_pth):
            shutil.rmtree(tmp_pth)
    # Temporary path
    tmp_pth = os.path.join(os.environ["HOME"], "tmp", "scrap")
    # Directory contents
    ld = os.listdir(pth)
    # Get directory's contents
    names = [fn for fn in ld
             if os.path.splitext(fn)[1] in model_exts + zip_exts]
    # Check that there is exactly one model file
    if len(names) != 1:
        return 1
    ## There is a unique model or zip file
    # Determine file name and extension, and unzip if necessary
    name, ext = gt.splitext2(os.path.basename(names[0]))
    if ext in zip_exts:
        # un-tar, un-zip into a temp directory
        untar(os.path.join(pth, name + ext), tmp_pth)
        pth = os.path.join(tmp_pth, name)
        # Directory contents
        ld = os.listdir(pth)
        # Get target's path
        names = [fn for fn in ld if os.path.splitext(fn)[1] in model_exts]
        # Check that there is exactly one model file
        if len(names) != 1:
            # Remove tmp path
            returnfunc()
            return 1
        name, ext = gt.splitext2(os.path.basename(names[0]))
    ## There is a unique model file
    if ext == ".obj":
        mtlname = name + ".mtl"
        # Is .mtl present?
        f_mtl = os.path.isfile(mtlname)
        if f_mtl:
            # Parse the .mtl file for the image names
            mtl_pth = os.path.join(pth, mtlname)
            mtlnames = parse_mtl_imgs(mtl_pth)
            # Directory for the images
            img_pth = os.path.join(pth, imgdirname)
            # Get image file names
            imgnames, imgpaths = parse_dir_imgs(img_pth)
            # Check that all .mtl img names are present
            if not set(mtlnames) <= set(imgnames):
                returnfunc()
                return 2
    returnfunc()
    return 0


def untar(tarname, tmp_pth):
    # Make the tmp_pth directory if it doesn't exist already
    if not os.path.isdir(tmp_pth):
        os.makedirs(tmp_pth)
    if gt.splitext2(tarname)[1] not in zip_exts:
        raise ValueError("Invalid zip file extension: %s" % tarname)
    try:
        # Open the tar
        with tarfile.open(tarname, 'r') as tf:
            # Extract it
            tf.extractall(tmp_pth)
            # Get tar.gz's member names
            names = tf.getnames()
    except:
        pdb.set_trace()
    return names

