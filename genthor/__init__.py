import urllib
import json

BASE_URL = 'http://50.19.109.25'
MODEL_URL = BASE_URL + ':9999/3dmodels?'

def get_canonical_view(m):
    v = json.loads(urllib.urlopen(MODEL_URL + 'query={"id":"' + m + '"}&fields=["canonical_view"]').read())[0]
    if v.get('canonical_view'):
        return v['canonical_view']