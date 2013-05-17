from pandac.PandaModules import *
import numpy as np
import genthor.datasets as gd; reload(gd)

def dist(o1, o2):
    preproc = {'dtype':'float32', 'size':(128, 128), 'normalize':False, 'mode':'L'}
    dataset = gd.GenerativeDatasetBase()
    fmap = dataset.imager.get_map(preproc, 'texture')
    N = 3
    poses = [(np.sin(2*np.pi*phi) * np.cos(2*np.pi*psi), np.sin(2*np.pi*phi) * np.sin(2*np.pi*psi),  np.cos(2*np.pi*phi)) for phi in np.arange(0, 1, 1./N) for psi in np.arange(0, 1, 1./N)]

    pdict1 = dist_rot(o1, 0, 0, 0, fmap, dataset, poses)
    M = 72
    rots = [(i, j, k) for i in range(360)[::M] for j in range(360)[::M] for k in range(360)[::M]]
    Ds = []
    for r in rots:
        pdict2 = dist_rot(o2, r[0], r[1], r[2], fmap, dataset, poses)
        Ds.append(np.mean([np.linalg.norm(pdict2[p] - pdict1[p]) for p in poses]))
    d = min(Ds)
    return d
    

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
    lbase = dataset.imager.renderers[('texture', (128, 128))][0]
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
    pdict = {}
    for p in poses:
        p_poses = [np.array(x) for x in P if is_collinear(p, x)][0]
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
    K = [float(a) / b if np.abs(b) > tol else np.abs(a) < tol  for a, b in zip(A, B) ]
    if False in K:
        return False
    else:
        K0 = [k for k in K if k != True]
        return all([np.abs((k - K0[0])) < tol for k in K0])


