""" Utilities."""

import numpy as np
import os
import boto

def read_file(func, filepth):
    """ Returns func(filepath), first trying absolute path, then
    relative."""

    try:
        out = func(filepth)
    except IOError:
        try:
            out = func(os.path.join(os.getcwd(), filepth))
        except IOError as exc:
            raise exc
    return out


def init_rand(rand):
    """ Takes either an existing mtrand.RandomState object, or a seed,
    and returns an mtrand.RandomState object."""
    
    # Random state
    if not isinstance(rand, np.random.mtrand.RandomState):
        # rand is a seed
        rand = np.random.RandomState(seed=rand)
    return rand


def sample(rng, num=1, f_log=False, rand=0):
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
        # add dimension to make operations broadcast properly
        arng = arng[:, None]

    # random values in [0, 1]
    rand = init_rand(rand)
    r = rand.rand(num, arng.shape[1])

    # fit them to the range
    val = r * (arng[[1]] - arng[[0]]) + arng[[0]]

    if f_log:
        # log-uniform
        val = np.exp(val)

    return val
    
    
def download_s3_directory(bucket_name, target):
    conn = boto.connect_s3()
    b = conn.get_bucket(bucket_name)
    L = list(b.list())
    L.sort(lambda x, y: x.name > y.name)
    L = L[::-1]
    f_suffix = '_$folder$'
    for l in L:
        n = l.name
        if n.endswith(f_suffix):
            dirname = n[:-len(f_suffix)]
            dir = os.path.join(target, dirname)
            if not os.path.exists(dir):
                print(n)
                os.mkdir(dir)
        else:
            filename = os.path.join(target, n)
            if not os.path.exists(filename):
                print(n)
                l.get_contents_to_filename(filename)

