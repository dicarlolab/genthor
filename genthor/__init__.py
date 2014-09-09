import os
import urllib
import json
import os
from skdata.data_home import get_data_home

S3_URL = "http://dicarlocox-datasets.s3.amazonaws.com"
s3_resource_bucket = 'genthor-resources'
s3_old_model_bucket = 'dicarlocox-3dmodels-v1'

BASE_NAME = 'genthor'
# Scikits data genthor directory
GENTHOR_PATH = os.path.join(get_data_home(), BASE_NAME)
# Resource root directory
RESOURCE_PATH = os.path.join(GENTHOR_PATH, "resources")
# Resource root directory
CACHE_PATH = os.path.join(GENTHOR_PATH, "cache")
# background root directory
BACKGROUND_PATH = os.path.join(RESOURCE_PATH, "backgrounds")
# .obj model root directory
OBJ_PATH = os.path.join(RESOURCE_PATH, "objs")
# .egg model root directory
EGG_PATH = os.path.join(RESOURCE_PATH, "eggs")
# .bam model root directory
BAM_PATH = os.path.join(RESOURCE_PATH, "bams")
HUMAN_PATH = os.path.join(RESOURCE_PATH, "human_data")
TEXTURE_PATH = os.path.join(RESOURCE_PATH, "textures")


def splitext2(pth, splitpoint=0):
    """ Better splitext than os.path's (take optional arg that lets
    you control which dot to split on)."""
    dirname, basename = os.path.split(pth)
    if "." in basename:
        splits = basename.split(".")
        n = len(splits)
        splitpoint = 0 if splitpoint < -n else splitpoint
        splitpoint = n - 1 if splitpoint >= n else splitpoint
        name = os.path.join(dirname, ".".join(splits[:splitpoint + 1]))
        ext = "." + ".".join(splits[splitpoint + 1:])
    else:
        name, ext = basename, ""
    return name, ext
