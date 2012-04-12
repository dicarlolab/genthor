#!/usr/bin/env python

""" Create a dataset of images and latent state descriptions, based on
models and backgrounds."""

import cPickle
import os
import sys
import numpy as np
from matplotlib.cbook import flatten
import genthor as gt
import genthor_renderer as gr
import pdb


def sample_model_bg(modelnames, bgnames, n_ex_per_model, n_ex):
    """ Samples a list of models and backgrounds for building the
    dataset. The models will be repeated 'n_ex_per_model' times.
    
    modelnames: list of model names
    bgnames: list of background filenames
    n_ex_per_model: number of examples per model
    n_ex: total number of examples

    modellist: list of 'n_ex' models
    bglist: list of 'n_ex' background names
    """

    # Make model list
    modellist = [m for m in flatten(zip(*([modelnames] * n_ex_per_model)))]
    # Make background list
    bgrand = np.random.randint(0, n_bg, n_ex)
    bglist = [bgnames[r] for r in bgrand]

    # Make category list
    # Model root directory
    model_path = os.path.join(os.environ["HOME"], "Dropbox/genthor/models/")
    # Get the model info that's contained in the scripts
    sys.path.append(model_path)
    model_categories = __import__("model_categories").MODEL_CATEGORIES
    # Assemble category info in dict with {modelname: category, ...}
    categories = []
    for categ, names in model_categories.iteritems():
        categories.extend([(name, categ) for name in names])
    categorydict = dict(categories)
    # The actual list
    categorylist = [categorydict[model] for model in modellist]
    
    return modellist, bglist, categorylist


def sample(rng, num=1, f_log=False):
    """ Samples 'num' random values in some range 'rng'.
    
    rng: range (can be either (2,) or (m,2)) 
    num: number of desired random samples
    f_log: log-uniform (good for scales)

    val: random values, shaped (m, num) (m=1 if rng is (2,))
    """
    
    # np.ndarray version of range
    arng = np.array(rng).T

    if f_log:
        if np.any(arng <= 0.):
            # error on non-positive log values
            raise ValueError("log is no good for non-positive values")
        # log-uniform
        arng = np.log(arng)

    if arng.ndim == 1:
        # add dimension to make operations broadcast right
        arng = arng[:, None]

    # random values in [0, 1]
    rand = np.random.rand(num, arng.shape[1])

    # fit them to the range
    val = rand * (arng[[1]] - arng[[0]]) + arng[[0]]

    if f_log:
        # log-uniform
        val = np.exp(val)

    return val


def latent2args(latent):
    """ Convert the 'latent' state to an 'args' tuple, suitable for
    rendering."""

    # Extract the values from 'latent'
    modelpath = gr.model_name2path(latent[0])
    bgpath = gr.bg_name2path(latent[1])
    category = latent[2]
    scale, pos, hpr, bgscale, bghp = latent[3:]

    # Put them into 'args'
    args = (modelpath, bgpath, scale, pos, hpr, bgscale, bghp)
    
    return args


def build_renderer_data(latents, out_root):
    """ Takes the 'latents' states and 'out_root' path, and returns
    the renderer 'all_args' arguments and the 'out_paths' output
    paths."""

    all_args = []
    out_paths = []
    for ilatent, latent in enumerate(latents):
        # all_args
        all_args.append(latent2args(latent))
        # out_paths
        filename = "scene%0*i" % (8, ilatent)
        out_paths.append(os.path.join(out_root, filename))

    return all_args, out_paths


######################################################################
# Set up the input for the renderer
#

# Bad backgrounds, do not use
bad_bgs = ['INTERIOR_13SN.hdr', 'INTERIOR_12SN.hdr', 'INTERIOR_11SN.hdr',
           'INTERIOR_10SN.hdr', 'INTERIOR_09SN.hdr', 'INTERIOR_08SN.hdr',
           'INTERIOR_07SN.hdr', 'INTERIOR_06SN.hdr', 'INTERIOR_05SN.hdr']

# Parameters of the models and background
n_categories = 11
n_model_per_category = 10
n_models = n_categories * n_model_per_category
n_bg = 136 - len(bad_bgs)
model_root = gt.MODEL_PATH
bg_root = gt.BACKGROUND_PATH


# Read modelnames and backgrounds from their root directories
modelnames = sorted(os.listdir(model_root))
bgnames = [bg for bg in sorted(os.listdir(bg_root)) if bg not in bad_bgs]
assert len(modelnames) == n_models, "number of models is wrong"
assert len(bgnames) == n_bg, "number of backgrounds is wrong"

# Parameters that will define how the dataset is made
n_ex_per_model = 100
n_ex = n_ex_per_model * n_models
image_size = (256, 256)

# Ranges for latent states
scale_rng = (0.5, 2.)
pos_rng = ((-1.0, 1.0), (-1.0, 1.0))
hpr_rng = ((-180., 180.), (-180., 180.), (-180., 180.))
bgscale_rng = (1.0, 1.0) #(0.5, 2.0)
bghp_rng = ((-180., 180.), (0., 0.))

# Make the 'n_ex' models and backgrounds list
modellist, bglist, categorylist  = sample_model_bg(
    modelnames, bgnames, n_ex_per_model, n_ex)

# Make the latent parameters lists
scalelist = sample(scale_rng, num=n_ex, f_log=True)
poslist = sample(pos_rng, num=n_ex)
hprlist = sample(hpr_rng, num=n_ex)
bgscalelist = sample(bgscale_rng, num=n_ex, f_log=True)
bghplist = sample(bghp_rng, num=n_ex)

# Make the list of latent states
latents = zip(modellist, bglist, categorylist, 
              scalelist, poslist, hprlist, bgscalelist, bghplist)

## We could make this a dict, but hold off for now
dictkeys = ("modelname", "bgname", "category", "scale", "pos", "hpr", "bgscale", "bghp")
latent_dicts = [dict(zip(dictkeys, latent)) for latent in latents]

# Build the 'all_args' and 'out_paths' lists, which will be fed to
# the rendering loop.
# 'out_root' points to the directory to place the images.
out_root = os.path.join(gt.GENTHOR_PATH, "images/training")
all_args, out_paths = build_renderer_data(latents, out_root)

######################################################################


######################################################################
# Renderer
#

# Here's the gist of running this thing to produce 10k images:
window_type = "onscreen" #"offscreen"

# Set up the renderer
lbase, output = gr.setup_renderer(window_type, size=image_size)
# if window_type == "offscreen":
#     tex = output.getTexture()
# Img = np.zeros((len(all_args), image_size[0], image_size[1], 3), "u1")

for args, out_path, latent_dict in zip(all_args, out_paths, latent_dicts):

    # Construct a scene
    objnode, bgnode = gr.construct_scene(lbase, *args)
    #modelpath, bgpath, scale, pos, hpr, bgscale, bghp = args
    # scenenode = gr.construct_scene(lbase, modelpath, bgpath, scale, pos,
    #                                hpr, bgscale, bghp)

    # Render the scene
    lbase.render_frame()

    # Remove of the constructed nodes
    objnode.removeNode()
    bgnode.removeNode()

    # Take a screenshot and save it
    lbase.screenshot(output, pth=out_path + ".jpg")

    # Save latent state
    with open(out_path + ".pkl", "w") as fid:
        cPickle.dump(latent_dict, fid)
    
    # if window_type == "offscreen":
    #     # Get the image
    #     img = lbase.get_tex_image(tex)

    # # Store the img
    # Img[iimg] = img

