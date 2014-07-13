"""
compute distances between two 3-d models via surface distortion mapping metric
"""
import math
import time
from pandac.PandaModules import (BitMask32,
                                 CollisionTraverser,
                                 CollisionNode,
                                 CollisionRay,
                                 CollisionHandlerQueue)

import numpy as np
import genthor.datasets as gd; reload(gd)


def surface_normals(config, preproc, delta):
    """
    preproc = {'dtype':'float32', 'size':(128, 128), 'normalize':False, 'mode':'L'}
    obj = '10_trumpet'
    config = {'bgname': 'DH-ITALY04SN.jpg',
                             'bgphi': 0,
                             'bgpsi': 0.0,
                             'bgscale': 1.0,
                             'obj': [obj],
                             'rxy': [0],
                             'rxz': [0],
                             'ryz': [0],
                             's': [1],
                             'tx': [0],
                             'ty': [0],
                             'tz': [0],
                             'texture': [None],
                             'texture_mode': [None],
                             'internal_canonical': True,
                             'use_envmap': False}
    """
    t = time.time()
    preproc = {'dtype':'float32', 'size':(128, 128), 'normalize':False, 'mode':'L'}
    dataset = gd.GenerativeDatasetBase()
    fmap = dataset.imager.get_map(preproc, 'texture')
    x = fmap(config)
    renderers = dataset.imager.renderers
    lbase = [renderers[_re] for _re in renderers if _re[:2] == ('texture', (128, 128))][0][0]

    camera_pos = tuple(lbase.cameras.getPos())
    lens = lbase.cameras.getChildren()[0].node().getLens()
    fov = np.radians(lens.getMinFov())
    z0 = camera_pos[1]
    x0 = camera_pos[0]
    y0 = camera_pos[2]
    sw = 2. * np.abs(z0) * np.tan(fov / 2.)
    rvals = np.arange(-sw/2., sw/2., delta)
    K = len(rvals)

    preproc = {'dtype':'float32', 'size':(K, K), 'normalize':False, 'mode':'L'}
    fmap = dataset.imager.get_map(preproc, 'texture')
    x = fmap(config, remove=False)
    lbase = [renderers[_re] for _re in renderers if _re[:2] == ('texture', (K, K))][0][0]
    root = lbase.rootnode
    c = list(root.getChildren())
    if len(c) > 2:
        while True:
            root.getChildren()[-1].removeNode()
            if len(list(root.getChildren())) <= 2:
                break
        x = fmap(config, remove=False)
    print('0', time.time() - t)
    queue = intersect(lbase, delta, mask=x)
    P = np.array([queue.getEntry(i).getSurfacePoint(root) for i in range(queue.getNumEntries())])
    def normalize(x):
        return x / np.linalg.norm(x)
    N = np.array([normalize(queue.getEntry(i).getSurfaceNormal(root)) for i in range(queue.getNumEntries())])
    D = np.linalg.norm(P - np.array(camera_pos), axis=1)
    sw = 2. * np.abs(z0) * np.tan(fov / 2.)
    rvals = np.arange(-sw/2., sw/2., delta)
    K = len(rvals)
    mat = np.zeros((K, K, 3), dtype=np.dtype('float32'))
    nm = np.linalg.norm(camera_pos)
    dmat = D.max() * np.ones((K, K), dtype=np.dtype('float32'))
    mapper = lambda x: (K/sw) * x + K/2
    for p, n, d in zip(P, N, D):
        assert p[1] >= z0, (p, z0)
        ratio = np.abs(z0) / np.abs(p[1] - z0)
        x1 = (p[0] - x0) * ratio
        y1 = (p[2] - y0) * ratio
        ival = int(round(mapper(x1)))
        jval = int(round(mapper(y1)))
        if dmat[ival, jval] > d:
            mat[ival, jval, :] = n
            dmat[ival, jval] = d
    root.getChildren()[2].removeNode()
    root.getChildren()[2].removeNode()
    root.getChildren()[2].removeNode()
    return mat, dmat


def intersect(lbase, delta, mask=None):
    t = time.time()
    root = lbase.rootnode
    terrain = root.getChildren()[2]
    terrain.setCollideMask(BitMask32.bit(1))
    player = root
    collTrav = CollisionTraverser() #the collision traverser. we need this to perform collide in the end
    fromObject = player.attachNewNode(CollisionNode('colNode'))

    camera_pos = tuple(lbase.cameras.getPos())

    lens = lbase.cameras.getChildren()[0].node().getLens()
    fov = np.radians(lens.getMinFov())
    zpos = camera_pos[1]
    sw = 2. * np.abs(zpos) * np.tan(fov / 2.)

    rvals = np.arange(-sw/2., sw/2., delta)
    if mask is None:
        poses = [(xpos , -1 * zpos, ypos) for xpos in rvals for ypos in rvals]
    else:
        assert mask.shape == (len(rvals), len(rvals))
        poses = [(xpos , -1 * zpos, ypos) for _i, xpos in enumerate(rvals) for _j, ypos in enumerate(rvals) if mask[_j, _i] < 1]
        print(len(poses), 'lp', len(rvals)**2)
    print('1a', time.time() - t)
    t = time.time()
    for p in poses:
        vec = camera_pos + p
        fromObject.node().addSolid(CollisionRay(*vec))
    print('1', time.time() - t)
    t = time.time()

    #and now we turn turn of the collision for the playermodel so it wont collide with the ray or anything.
    player.node().setIntoCollideMask(BitMask32.allOff())
    fromObject.node().setFromCollideMask(BitMask32.bit(1))

    queue = CollisionHandlerQueue()
    collTrav.addCollider(fromObject, queue)
    print('2', time.time() - t)
    t = time.time()

    collTrav.traverse(root)
    print('3', time.time() - t)
    return queue



