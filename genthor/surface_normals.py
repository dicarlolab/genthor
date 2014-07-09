"""
compute distances between two 3-d models via surface distortion mapping metric
"""

from pandac.PandaModules import (BitMask32,
                                 CollisionTraverser,
                                 CollisionNode,
                                 CollisionRay,
                                 CollisionHandlerQueue)

import numpy as np
import genthor.datasets as gd; reload(gd)


def dist_rot(config, preproc, delta):
    """
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

    preproc = {'dtype':'float32', 'size':(128, 128), 'normalize':False, 'mode':'L'}
    dataset = gd.GenerativeDatasetBase()
    fmap = dataset.imager.get_map(preproc, 'texture')
    x = fmap(config, remove=False)
    lbase = dataset.imager.renderers[('texture', (128, 128))][0]
    root = lbase.rootnode
    c = list(root.getChildren())
    if len(c) > 2:
        while True:
            root.getChildren()[-1].removeNode()
            if len(list(root.getChildren())) <= 2:
                break
        x = fmap(config, remove=False)

    queue = intersect(root, delta)

    P = [np.array((queue.getEntry(i).getSurfacePoint(root)) for i in range(queue.getNumEntries())]
    N = [np.array((queue.getEntry(i).getSurfaceNormal(root)) for i in range(queue.getNumEntries())]

    camera_pos = tuple(lbase.cameras.getPos())
    lens = lbase.cameras.getChildren()[0].node().getLens()
    fov = lens.getMinFov()
    z0 = camera_pos[1]
    x0 = camera_pos[0]
    y0 = camera_pos[2]

    scene_width = 2 * np.abs(zpos) * np.tan(fov / 2)
    rvals = np.arange(-sw/2, sw/2, delta)
    K = len(rvals)

    mat = np.zeros(K, K, 3)
    for p, n in zip(P, N):
        assert p[1] >= z0, (p, z0)
        ratio = np.abs(z0) / np.abs(p[1] - z0)
        x1 = (p[0] - x0) * ratio
        y1 = (p[2] - y0) * ratio
        ival = int(math.ceil((2 * x1 / scene_width) + 1))
        jval = int(math.ceil((2 * xy / scene_width) + 1))
        mat[ival, jval, :] = n

    root.getChildren()[2].removeNode()
    root.getChildren()[2].removeNode()
    root.getChildren()[2].removeNode()

    return mat


def intersect(root, delta):
    terrain = root.getChildren()[2]
    terrain.setCollideMask(BitMask32.bit(1))
    player = root
    collTrav = CollisionTraverser() #the collision traverser. we need this to perform collide in the end
    fromObject = player.attachNewNode(CollisionNode('colNode'))

    camera_pos = tuple(lbase.cameras.getPos())

    lens = lbase.cameras.getChildren()[0].node().getLens()
    fov = lens.getMinFov()
    zpos = camera_pos[1]
    scene_width = 2 * np.abs(zpos) * np.tan(fov / 2)

    rvals = np.arange(-sw/2, sw/2, delta)
    poses = [(xpos , -1 * zpos, ypos) for xpos in rvals for ypos in ravls]

    for p in poses:
        vec = camera_pos + p
        fromObject.node().addSolid(CollisionRay(*p))

    #and now we turn turn of the collision for the playermodel so it wont collide with the ray or anything.
    player.node().setIntoCollideMask(BitMask32.allOff())
    fromObject.node().setFromCollideMask(BitMask32.bit(1))

    queue = CollisionHandlerQueue()
    collTrav.addCollider(fromObject, queue)

    collTrav.traverse(root)
    return queue



