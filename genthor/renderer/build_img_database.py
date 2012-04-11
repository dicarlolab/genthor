#!/usr/bin/env python

""" Create a dataset of images and latent state descriptions, based on
models and backgrounds."""

import collections
import os
import re
import sys
import numpy as np
from matplotlib.cbook import flatten
import genthor as gt
import genthor_renderer as gr
import pdb


# TODO: make scale & bgscale 1D, pos 2D, original instance name, +
# category name
# - verify long width is length 1
# - camera should be (-1.5, 1.5)



def sample_model_bg(modelnames, bgnames, n_examples_per_model, n_examples):
    """ Samples a list of models and backgrounds for building the
    dataset. The models will be repeated 'n_examples_per_model' times.
    
    modelnames: list of model names
    bgnames: list of background filenames
    n_examples_per_model: number of examples per model
    n_examples: total number of examples

    modellist: list of 'n_examples' models
    bglist: list of 'n_examples' background names
    """

    modellist = [m for m in flatten(zip(*([modelnames] * n_examples_per_model)))]
    bgrand = np.random.randint(0, n_bg, n_examples)
    bglist = [bgnames[r] for r in bgrand]

    return modellist, bglist


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
    scale, pos, hpr, bgscale, bghp = latent[2:]

    # Put them into 'args'
    args = (modelpath, bgpath, scale, pos, hpr, bgscale, bghp)
    
    return args


def build_renderer_data(latents, out_root):
    """ Takes the 'latents' states and 'out_root' path, and returns
    the renderer 'all_args' arguments and the 'out_paths' output
    paths."""

    pat = re.compile("\D+")
    all_args = []
    out_paths = []
    for ilatent, latent in enumerate(latents):
        # all_args
        all_args.append(latent2args(latent))
        # out_paths
        category = pat.findall(latent[0])[0]
        filename = "scene%0*i" % (8, ilatent)
        out_paths.append(os.path.join(out_root, filename))

    return all_args, out_paths


######################################################################
# Set up the input for the renderer
#

# Parameters of the models and background
n_categories = 11
n_model_per_category = 10
n_models = n_categories * n_model_per_category
n_bg = 136
model_root = gt.MODEL_PATH
bg_root = gt.BACKGROUND_PATH

# Read modelnames and backgrounds from their root directories
modelnames = sorted(os.listdir(model_root))
bgnames = sorted(os.listdir(bg_root))
assert len(modelnames) == n_models, "number of models is wrong"
assert len(bgnames) == n_bg, "number of backgrounds is wrong"

# Parameters that will define how the dataset is made
n_examples_per_model = 100
n_examples = n_examples_per_model * n_models
image_size = (512, 512)

# Ranges for latent states
scale_rng = (0.5, 2.)
pos_rng = ((-1.0, 1.0), (-1.0, 1.0))
hpr_rng = ((-180., 180.), (-180., 180.), (-180., 180.))
bgscale_rng = (0.5, 2.0)
bghp_rng = ((-180., 180.), (-180., 180.))

# Make the 'n_examples' models and backgrounds list
modellist, bglist = sample_model_bg(
    modelnames, bgnames, n_examples_per_model, n_examples)

# Make the latent parameters lists
scalelist = sample(scale_rng, num=n_examples, f_log=True)
poslist = sample(pos_rng, num=n_examples)
hprlist = sample(hpr_rng, num=n_examples)
bgscalelist = sample(bgscale_rng, num=n_examples, f_log=True)
bghplist = sample(bghp_rng, num=n_examples)

# Make the list of latent states
latents = zip(modellist, bglist,
              scalelist, poslist, hprlist, bgscalelist, bghplist)

## We could make this a dict, but hold off for now
# dictkeys = ("modelname", "bgname", "scale", "pos", "hpr", "bgscale", "bghp")
# latents_dict = [dict(zip(dictkeys, latent)) for latent in latents]

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
window_type = "offscreen"

# Set up the renderer
lbase, output = gr.setup_renderer(window_type)
# if window_type == "offscreen":
#     tex = output.getTexture()
# Img = np.zeros((len(all_args), 512, 512, 3), "u1")

for iimg, (args, out_path) in enumerate(zip(all_args, out_paths)[:3]):

    # Construct a scene
    scenenode = gr.construct_scene(lbase, *args)
    #modelpath, bgpath, scale, pos, hpr, bgscale, bghp = args
    # scenenode = gr.construct_scene(lbase, modelpath, bgpath, scale, pos,
    #                                hpr, bgscale, bghp)

    # Render the scene
    lbase.render_frame()

    # Take a screenshot and save it
    lbase.screenshot(output, pth=out_path)
    
    # if window_type == "offscreen":
    #     # Get the image
    #     img = lbase.get_tex_image(tex)

    # # Store the img
    # Img[iimg] = img

