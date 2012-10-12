""" General tools for manipulating models."""
import genthor as gt
import os
import re
import shutil
import tarfile


# Model extensions
model_exts = (".obj", ".egg", ".bam")
# Zip extensions
zip_exts = (".tgz", ".tar.gz", ".tbz2", ".tar.bz2")
# Texture image file extensions
img_exts = (".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".png")
# .mtl image fields
mtl_img_fields = ("map_Ka", "map_Kd", "map_bump", "bump", "map_refl")


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



# def fix_tex_names(mtl_path, imgdirname="tex", f_verify=True):
#     """ Make all .mtl image file names lowercase and relative paths,
#     so they are compatible with linux and are portable.  Also change
#     the actual image file names."""

#     # Texture image file extensions
#     img_exts = (".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".png")

#     # .mtl image fields
#     mtl_img_fields = ("map_Ka", "map_Kd", "map_bump", "bump", "map_refl")

#     # Directory path that the .mtl file is in
#     dir_path = os.path.split(mtl_path)[0]

#     # Name of this directory
#     dir_name = os.path.split(dir_path)[1]

#     # Directory for the images
#     img_pth = os.path.join(dir_path, imgdirname)

#     # visit() function for os.path.walk().  It appends detected image
#     # files to a list.
#     def visit(imgnames, dir_pth, names):
#         imgnames.extend([os.path.join(dir_pth, name) for name in names
#                          if os.path.splitext(name)[1].lower() in img_exts])
#     # walk down directory tree and get the image files    
#     imgpaths0 = []
#     for dp, foo, names in os.walk(dir_path):
#         visit(imgpaths0, dp, names)
#     imgnames = [os.path.split(pth)[1].lower() for pth in imgpaths0]

#     # RE pattern
#     pat_fields = "(?:" + "|".join(mtl_img_fields) + ") "
#     pat_img_exts = "(.+(?:\\" + "|\\".join(img_exts) + "))"
#     patstr = "[\s]*" + pat_fields + "((?:.*[/\\\\])?" + pat_img_exts + ")"
#     rx = re.compile(patstr, re.IGNORECASE)

#     # Initialize storage for the image file names inside the .mtl
#     mtlnames = []
#     mtllines = []
   
#     ## Get the image file names from the .mtl file
#     # Open .mtl
#     with open(mtl_path, "r") as fid:
#         # Iterate over lines
#         for line in fid.readlines():
#             # Search the line
#             m = rx.search(line)
#             if m is not None:
#                 # Pull out the path and image name
#                 #pth = m.group(1)
#                 Name = m.group(2)
#                 name = Name.lower()
#                 # If an image file name is found, store it
#                 mtlnames.append(name)
#                 # Edit the line and store
#                 newline = (line[:m.start(1)] + os.path.join(imgdirname, name)
#                            + line[m.end(1):])
#                 mtllines.append(newline)
#             else:
#                 mtllines.append(line)

#     ## Edit .mtl files
#     # Open .mtl
#     with open(mtl_path, "w") as fid:
#         # Iterate over lines
#         for line in mtllines:
#             # Write the line
#             fid.write(line)
                
#     # Make unique and sort
#     mtlnames = sorted(set(mtlnames))

#     if f_verify:
#         # Verify that all the mtl images are present
#         for mtlname in mtlnames:
#             if mtlname not in imgnames:
#                 raise ValueError("Cannot find .mtl-defined image. "
#                                  "mtl: %s. img: %s" % (mtl_path, mtlname))

#     # Make the directory if need be, and error if it is a file already
#     if os.path.isfile(img_pth):
#         raise IOError("File exists: '%s'")
#     elif not os.path.isdir(img_pth):
#         # Make image directory, if necessary
#         os.makedirs(img_pth)
    
#     # Move the image files to the new img_pth location
#     for imgpath0, imgname in zip(imgpaths0, imgnames):
#         imgpath = os.path.join(img_pth, imgname)
#         shutil.move(imgpath0, imgpath)
#         #print "%s --> %s" % (imgpath0, imgpath)


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

