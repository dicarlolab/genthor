#!/usr/bin/env python

""" Create a dataset of images and state descriptions, based on models
and backgrounds."""

import cPickle
import os
import sys
from collections import OrderedDict as dict
import numpy as np
from matplotlib.cbook import flatten
import genthor as gt
import renderer as gr
import genthor.tools as tools
import pdb


def sample_model_bg(modelnames, bgnames, n_ex_per_model, n_ex, rand=0):
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
    rand = tools.init_rand(rand)
    bgrand = rand.randint(0, n_bg, n_ex)
    bglist = [bgnames[r] for r in bgrand]

    # Make category list
    # Model root directory
    model_path = os.path.join(gt.GENTHOR_PATH, "models")
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


def state2args(state):
    """ Convert the 'state' state to an 'args' tuple, suitable for
    rendering."""

    # Extract the values from 'state'
    modelpath = gr.model_name2path(state[0])
    bgpath = gr.bg_name2path(state[1])
    category = state[2]
    scale, pos, hpr, bgscale, bghp = state[3:]

    # Put them into 'args'
    args = (modelpath, bgpath, scale, pos, hpr, bgscale, bghp)
    
    return args


def build_renderer_data(states, out_root):
    """ Takes the 'states' states and 'out_root' path, and returns
    the renderer 'all_args' arguments and the 'out_paths' output
    paths."""

    if not os.path.exists(out_root):
        os.mkdir(out_root)

    all_args = []
    out_paths = []
    for istate, state in enumerate(states):
        # all_args
        all_args.append(state2args(state))
        # out_paths
        filename = "scene%0*i" % (8, istate)
        out_paths.append(os.path.join(out_root, filename))

    return all_args, out_paths


######################################################################
# Set up the input for the renderer
#

# Bad backgrounds, do not use
bad_bgs = ['INTERIOR_13ST.jpg', 'INTERIOR_12ST.jpg', 'INTERIOR_11ST.jpg',
           'INTERIOR_10ST.jpg', 'INTERIOR_09ST.jpg', 'INTERIOR_08ST.jpg',
           'INTERIOR_07ST.jpg', 'INTERIOR_06ST.jpg', 'INTERIOR_05ST.jpg']

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
assert len(modelnames) == n_models, ("Wrong number of models:"
                                     "len(modelnames) != n_models (%i != %i)"
                                     % (len(modelnames), n_models))
assert len(bgnames) == n_bg, ("Wrong number of backgrounds:"
                              "len(bgnames) != n_bg (%i != %i)"
                              % (len(bgnames), n_bg))

# Parameters that will define how the dataset is made
n_ex_per_model = 100
n_ex = n_ex_per_model * n_models
image_size = (256, 256)

# Ranges for states
scale_rng = (0.6667, 2.)
pos_rng = ((-1.0, 1.0), (-1.0, 1.0))
hpr_rng = ((-180., 180.), (-180., 180.), (-180., 180.))
bgscale_rng = (1.0, 1.0) #(0.5, 2.0)
bghp_rng = ((-180., 180.), (0., 0.))

# Make the 'n_ex' models and backgrounds list
modellist, bglist, categorylist  = sample_model_bg(
    modelnames, bgnames, n_ex_per_model, n_ex)

# Make the state parameters lists
sample = tools.sample
scalelist = sample(scale_rng, num=n_ex, f_log=True)
poslist = sample(pos_rng, num=n_ex)
hprlist = sample(hpr_rng, num=n_ex)
bgscalelist = sample(bgscale_rng, num=n_ex, f_log=True)
bghplist = sample(bghp_rng, num=n_ex)

# Make the list of states
states = zip(modellist, bglist, categorylist, 
              scalelist, poslist, hprlist, bgscalelist, bghplist)

## We could make this a dict, but hold off for now
dictkeys = ("modelname", "bgname", "category", "scale", "pos", "hpr",
            "bgscale", "bghp")
state_dicts = [dict(zip(dictkeys, state)) for state in states]

# Build the 'all_args' and 'out_paths' lists, which will be fed to
# the rendering loop.
# 'out_root' points to the directory to place the images.
out_root = os.path.join(gt.GENTHOR_PATH, "training_data")
all_args, out_paths = build_renderer_data(states, out_root)

######################################################################


######################################################################
# Renderer
#

# Here's the gist of running this thing to produce 10k images:
window_type = "onscreen" #"offscreen"

# Set up the renderer
lbase, output = gr.setup_renderer(window_type, size=image_size)

for args, out_path, state_dict in zip(all_args, out_paths, state_dicts):

    # Construct a scene
    objnode, bgnode = gr.construct_scene(lbase, *args)

    # Render the scene
    lbase.render_frame()

    # Remove of the constructed nodes
    objnode.removeNode()
    bgnode.removeNode()

    # Take a screenshot and save it
    lbase.screenshot(output, pth=out_path + ".jpg")

    # Save state
    with open(out_path + ".pkl", "w") as fid:
        cPickle.dump(state_dict, fid)
