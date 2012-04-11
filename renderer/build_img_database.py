#!/usr/bin/env python

import os
import sys
import numpy as np
import genthor_renderer as gr
import pdb


# Here's the gist of running this thing to produce 10k images:
window_type = "offscreen"

# Set up the renderer
lbase, output = setup_renderer(window_type)
if window_type == "offscreen":
    tex = output.getTexture()

# Fill with the arguments for each img you want rendered.
all_args = []

Img = np.zeros((len(all_args), 512, 512, 3), "u1")

for iimg, args in enumerate(all_args):

    # Construct a scene
    modelpath, envpth, scale, pos, hpr, phitheta, bgscale = args
    scenenode = construct_scene(lbase, modelpath, envpth, scale, pos,
                                hpr, phitheta, bgscale)

    # Render the scene
    lbase.render_frame()

    if window_type == "offscreen":
        # Get the image
        img = lbase.get_tex_image(tex)

    # Store the img
    Img[iimg] = img

# Save out images, or probably do it in the for loop if there are
# more than a few hundred.
