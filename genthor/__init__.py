import os
import urllib
import json
import os

BASE_URL = 'http://50.19.109.25'
MODEL_URL = BASE_URL + ':9999/3dmodels?'

# Root genthor project path
GENTHOR_PATH = os.path.abspath(os.environ["GENTHOR_PATH"])
# Points to the models
MODEL_PATH = os.path.join(GENTHOR_PATH, "processed_models")
# Points to the backgrounds
BACKGROUND_PATH = os.path.join(GENTHOR_PATH, "backgrounds")

def get_canonical_view(m):
    v = json.loads(urllib.urlopen(MODEL_URL + 'query={"id":"' + m + '"}&fields=["canonical_view"]').read())[0]
    if v.get('canonical_view'):
        return v['canonical_view']
