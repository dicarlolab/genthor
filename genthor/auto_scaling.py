import genthor
import genthor.datasets as gd
from boto.s3.connection import S3Connection
import numpy

##Connect to S3 and get list of object names.
def get_canonical_list():
    obj_list = []
    conn = S3Connection()  
    bucket = conn.get_bucket('dicarlocox-3dmodels-v1')
    for key in bucket:
        obj_list.append(key.name)
    return obj_list

   
def make_canonical(obj_list):
    dataset = gd.GenerativeDatasetBase()
    cb = gd.CanonicalBase(dbname='scaletest',colname='autoset')
    preproc = {'dtype':'float32', 'size':(250, 250, 3), 'normalize':False, 'mode':'RGB'}
    irr = dataset.imager.get_map(preproc, 'texture')
    lbase = dataset.imager.renderers[('texture', (250, 250, 3))][0]
    for obj_name in obj_list: 
	if not isinstance(cb.getCanonical(str(obj_name)[:len(obj_name)-7],'dyamins-auto'), dict):
		try:
			config = {'bgname': 'MOUNT_24SN.jpg',
			 'bgphi': 17.572861413836904,
			 'bgpsi': 0.0,
			 'bgscale': 1.0,
			 'obj': '',
			 'rxy': 0.0,
			 'rxz': 0.0,
			 'ryz': 0.0,
			 's': 1.0,
			 'tx': 0.0,
			 'ty': 0.0,
			 'tz': 0.0,
			 'texture': None, 
			 'texture_mode': None,
			 'user':'dyamins-auto'}
			 
			config['obj'] = str(obj_name)[:len(obj_name)-7]			
			download = dataset.get_image(preproc,config);	#Need to ensure obj is downloaded		
			x = irr(config, remove=False)
			vertices = numpy.array(lbase.rootnode.getChildren()[2].getTightBounds())

			#Size across full image is 3, so size across 40% is always 1.2 
			initial_scale_factor = max(abs(vertices[0]-vertices[1]))
			canonical_scale = 1.2/initial_scale_factor
                        config['s'] = canonical_scale
		
			#canonical position sets position to 0
			cpos = vertices.mean(0)
			config['tx'] = -cpos[1]
			config['ty'] = cpos[0]
			config['tz'] = cpos[2]
                        
			print(config)
				
			cb.saveCanonical(preproc, config)    
		
			lbase.rootnode.getChildren()[2].removeNode()
			lbase.rootnode.getChildren()[2].removeNode()
		
			print(obj_name)
		except (IOError, ValueError, AssertionError):
			print('Failed to load model '+str(obj_name))
			continue
		
	else:
		print('Model '+str(obj_name)+' already scaled. Moving on...')
		continue
