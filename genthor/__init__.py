import os
import urllib
import json
import os
from skdata.data_home import get_data_home

BASE_URL = 'http://50.19.109.25'
MODEL_URL = BASE_URL + ':9999/3dmodels?'

# Scikits data genthor directory
SKDATA_PATH = os.path.join(get_data_home(), "genthor")
# Resource root directory
RESOURCE_PATH = os.path.join(SKDATA_PATH, "resources")
# Resource root directory
CACHE_PATH = os.path.join(SKDATA_PATH, "cache")
# background root directory
BACKGROUND_PATH = os.path.join(RESOURCE_PATH, "backgrounds")
# .obj model root directory
OBJ_PATH = os.path.join(RESOURCE_PATH, "objs")
# .egg model root directory
EGG_PATH = os.path.join(RESOURCE_PATH, "eggs")
# .bam model root directory
BAM_PATH = os.path.join(RESOURCE_PATH, "bams")


def get_canonical_view(m):
    v = json.loads(urllib.urlopen(MODEL_URL + 'query={"id":"' + m + '"}&fields=["canonical_view"]').read())[0]
    if v.get('canonical_view'):
        return v['canonical_view']


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
