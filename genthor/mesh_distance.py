
import os
import cPickle

"""
compute distances between two 3-d models via surface distortion mapping metric
"""


from pandac.PandaModules import *
import numpy as np
import genthor.datasets as gd; reload(gd)


class PdictError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr('No pdict for %s' % value)
    
    
def dist(o1, o2, write=False, outdir=None, N=3.):
    """
    returns distance between two 3-d models o1 and o2
        example:
             >>>d = md.dist('face0001', ['schnauzer'])
             >>>d
                {'schnauzer': 0.26658963019761267}
    """
    preproc = {'dtype':'float32', 'size':(128, 128), 'normalize':False, 'mode':'L'}
    dataset = gd.GenerativeDatasetBase()
    fmap = dataset.imager.get_map(preproc, 'texture')

    _N = 0
    while True:
        poses = [(np.sin(2*np.pi*phi) * np.cos(2*np.pi*psi), np.sin(2*np.pi*phi) * np.sin(2*np.pi*psi),  np.cos(2*np.pi*phi)) for phi in np.arange(0, 1, 1./N) for psi in np.arange(0, 1, 1./N)]

        pdict1 = dist_rot(o1, 0, 0, 0, fmap, dataset, poses)
        if not pdict1:
            if _N < 3:
                _N += 1
                N += 2
            else:
                print('No pdict1')
                return {}
        else:
            break
        
    M = 72
    rots = [(i, j, k) for i in range(360)[::M] for j in range(360)[::M] for k in range(360)[::M]]
    dist_dict = {}
    for _o2 in o2:
        print(o1, _o2)
        try:
            Ds = []
            for r in rots:
                pdict2 = dist_rot(_o2, r[0], r[1], r[2], fmap, dataset, poses)
                if pdict2:
                    Ds.append(np.mean([np.linalg.norm(pdict2[p] - pdict1[p]) for p in poses]))
            d = min(Ds)
            if write:
                pth = os.path.join(outdir, o1 + '_' + _o2 + '.pkl')
                with open(pth, 'w') as _f:
                    cPickle.dump(d, _f)
            dist_dict[_o2] = d
        except (AssertionError, ValueError), e:
            print(o1, _o2, 'failure', e)
    return dist_dict


def dist_rot(obj, rxy, rxz, ryz, fmap, dataset, poses):
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

    x = fmap(config, remove=False)
    rk = [_k for _k in dataset.imager.renderers if _k[:2] == ('texture', (128, 128))][0]
    lbase = dataset.imager.renderers[rk][0]
    root = lbase.rootnode
    c = list(root.getChildren())
    if len(c) > 2:
        while True:
            root.getChildren()[-1].removeNode()
            if len(list(root.getChildren())) <= 2:
                break
        x = fmap(config, remove=False)
    queue = thing(root, poses)

    P = [list(queue.getEntry(i).getSurfacePoint(root)) for i in range(queue.getNumEntries())]
    if not P:
        return {}

    pdict = {}
    for p in poses:
        cfs = np.array([1 - np.corrcoef(x, p)[0, 1] for x in P])
        p_poses = np.array(P[cfs.argmin()])
        #p_poses = [np.array(x) for x in P if is_collinear(p, x)][0]
        pdict[p] = p_poses

    root.getChildren()[2].removeNode()
    root.getChildren()[2].removeNode()
    root.getChildren()[2].removeNode()

    return pdict


def thing(root,poses):
    terrain = root.getChildren()[2]
    terrain.setCollideMask(BitMask32.bit(1))
    player = root
    collTrav = CollisionTraverser() #the collision traverser. we need this to perform collide in the end
    fromObject = player.attachNewNode(CollisionNode('colNode'))
    for p in poses:
        fromObject.node().addSolid(CollisionRay(0, 0, 0, p[0], p[1], p[2]))

    #and now we turn turn of the collision for the playermodel so it wont collide with the ray or anything.
    player.node().setIntoCollideMask(BitMask32.allOff())
    fromObject.node().setFromCollideMask(BitMask32.bit(1))

    queue = CollisionHandlerQueue()
    collTrav.addCollider(fromObject, queue)

    collTrav.traverse(root)
    return queue


def is_collinear(A, B):
    tol = 1e-3
    K = [float(a) / b if np.abs(b) > tol else (True if np.abs(a) < tol else 'F')  for a, b in zip(A, B) ]
    if 'F' in K:
        return False
    else:
        K0 = [k for k in K if k != True]
        tol_list = [np.abs((k - K0[0])) for k in K0]
        A = all([t < tol for t in tol_list])
        if A:
            return True
        else:
            print(tol_list)
            return False

