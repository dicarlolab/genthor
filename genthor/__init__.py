import urllib
import json
import os

BASE_URL = 'http://50.19.109.25'
MODEL_URL = BASE_URL + ':9999/3dmodels?'


# The GENTHOR env variable points to the base directory of the project.
if 'GENTHOR' not in os.environ:
    # Put GENTHOR in your .bashrc or whatever.
    raise EnvironmentError('You must define the environment variable: GENTHOR') 
else:
    ROOT_PATH = os.path.abspath(os.environ["GENTHOR"])


def get_canonical_view(m):
    v = json.loads(urllib.urlopen(MODEL_URL + 'query={"id":"' + m + '"}&fields=["canonical_view"]').read())[0]
    if v.get('canonical_view'):
        return v['canonical_view']
