#!/usr/bin/python

import os
import sys
import shutil
import subprocess

import obj2egg



blender_script_name = 'obj2egg.py'


blender_command_base = 'blender -b -P %s -- ' % blender_script_name




# Append the model directory to the path
sys.path.append(MODEL_PATH)

# Get the model info that's contained in the scripts
model_categories_script = 'model_categories.py'
canonical_angles_script = 'canonical_angles.py'

try:
    model_categories = __import__(model_categories_script)
    canonical_angles = __import__(canonical_angles_script)

    for i in config.__dict__:
        print i            
except ImportError:
    print "Unable to import configuration file %s" % (myconfigfile,)

