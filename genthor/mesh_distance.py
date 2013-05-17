from pandac.PandaModules import *
import genthor.datasets as gd; reload(gd)

def dist(o1, o2):
    preproc = {'dtype':'float32', 'size':(128, 128), 'normalize':False, 'mode':'l'}
    fmap = dataset.imager.get_map(preproc, 'texture')
    N = 3
    poses = [(np.sin(2*np.pi*phi) * np.cos(2*np.pi*psi), np.sin(2*np.pi*phi) * np.sin(2*np.pi*psi),  np.cos(2*np.pi*phi)) for phi in np.arange(0, 1, 1./N) for psi in np.arange(0, 1, 1./N)]
    
    dataset = gd.GenerativeBase()    
    pdict1 = dist_rot(o1, 0, 0, 0, fmap, dataset, poses)
    M = 180
    rots = [(i, j, k) for i in range(360)[::M] for j in range(360)[::M] for k in range(360)[::M]]
    Ds = []
    for r in rots:
        pdict2 = dist_rot(o2, r[0], r[1], r[2], fmap, dataset, poses)
        Ds.append(np.mean([(pdict2[p] - pdict1[p])**2 for p in poses]))
    d = min(Ds)
    return d
    

def dist_rot(obj, rxy, rxz, ryz, fmap, dataset, poses):
    config = {'bgname': None,
                             'bgphi': 0,
                             'bgpsi': 0.0,
                             'bgscale': 1.0,
                             'obj': [obj],
                             'rxy': [rxy],
                             'rxz': [rxz],
                             'ryz': [ryz],
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
    #load and setup out test environment
    #terrain = lbase.loader.loadModel("smiley")
    #terrain.reparentTo(root)
    terrain = root.getChildren()[2]
    terrain.setCollideMask(BitMask32.bit(1))
    
    #load some player model
    player = lbase.loader.loadModel("smiley")
    player.setScale(.0001)
    player.reparentTo(root)
    #player = root
    
    collTrav = CollisionTraverser() #the collision traverser. we need this to perform collide in the end
    
    #our "from" node which is able to bump into "into" objects gets attached to the player
    fromObject = player.attachNewNode(CollisionNode('colNode'))
    #the actual solid which will be used as "from" object. in this case a ray pointing downwards from the players center

    for p in poses:
        fromObject.node().addSolid(CollisionRay(0, 0, 0) + p)
    #and now we turn turn of the collision for the playermodel so it wont collide with the ray or anything.
    player.node().setIntoCollideMask(BitMask32.allOff())
    #now set the bitmask of the "from" object to match the bitmask of the "into" object
    fromObject.node().setFromCollideMask(BitMask32.bit(1))
    
    #setting up a collision handler queue which will collect the collisions in a list
    queue = CollisionHandlerQueue()
    #add the from object to the queue so its collisions will get listed in it.
    collTrav.addCollider(fromObject, queue)
    
    collTrav.traverse(root)
    
    #just print the foudn collisions into the terminal so you can see what information the collision queue contains
    
    P = [list(queue.getEntry(i).getSurfacePoint(root)) for i in range(queue.getNumEntries())]
    
    pdict = {}
    for p in poses:
        p_poses = [x for x in poses if is_collinear(p, x)][0]
        pdict[p] = p_poses
        
    root.getChildren()[2].removeNode()
    root.getChildren()[2].removeNode()
    
    return pdict