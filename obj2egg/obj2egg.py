import bpy, os, sys, pdb, mathutils

""" Usage: 
$ blender -b -P obj2meshdata.py -- <obj filename>

"""


def importObj(pth):
    bpy.ops.import_scene.obj(filepath=pth)


def exportEgg(pth):
    import io_scene_egg.yabee_libs.egg_writer
    #from io_scene_egg.yabee_libs import egg_writer
    print('RELOADING MODULES')
    import imp
    imp.reload(io_scene_egg.yabee_libs.egg_writer)

    #: { 'animation_name' : (start_frame, end_frame, frame_rate) }
    ANIMATIONS = {'anim1':(0,10,5), }
    #: 'True' to interprete an image in the uv layer as the texture
    EXPORT_UV_IMAGE_AS_TEXTURE = False 
    #: 'True' to copy texture images together with main.egg
    COPY_TEX_FILES = True
    #: Path for the copied textures. Relative to the main EGG file dir.
    #: For example if main file path is '/home/username/test/test.egg',
    #: texture path is './tex', then the actual texture path is 
    #: '/home/username/test/tex'
    TEX_PATH = './tex'
    #: 'True' to write an animation data into the separate files
    SEPARATE_ANIM_FILE = True
    #: 'True' to write only animation data
    ANIM_ONLY = False
    #: number of sign after point
    FLOATING_POINT_ACCURACY = 4
    #: Enable tangent space calculation. Tangent space needed for some 
    # shaders/autoshaders, but increase exporting time
    # 'NO', 'INTERNAL', 'PANDA'
    # 'INTERNAL' - use internal TBS calculation
    # 'PANDA' - use egg-trans to calculate TBS
    # 'NO' - do not calc TBS
    CALC_TBS = 'PANDA'#'NO' #
    #: Type of texture processing. May be 'SIMPLE' or 'BAKE'.
    # 'SIMPLE' - export all texture layers as MODULATE. 
    # Exceptions: 
    #   use map normal == NORMAL
    #   use map specular == GLOSS
    #   use map emit == GLOW
    # 'BAKE' - bake textures. BAKE_LAYERS setting up what will be baked.
    # Also diffuse color of the material would set to (1,1,1) in the 
    # 'BAKE' mode
    #TEXTURE_PROCESSOR = 'BAKE'
    TEXTURE_PROCESSOR = 'SIMPLE'
    # type: (size, do_bake)
    BAKE_LAYERS = {'diffuse':(512, True),
                   'normal':(512, True),
                   'gloss': (512, True),    # specular
                   'glow': (512, False)      # emission
                   }

    egg_writer = io_scene_egg.yabee_libs.egg_writer
    egg_writer.write_out(pth, 
                        ANIMATIONS,
                        EXPORT_UV_IMAGE_AS_TEXTURE, 
                        SEPARATE_ANIM_FILE, 
                        ANIM_ONLY,
                        COPY_TEX_FILES, 
                        TEX_PATH, 
                        FLOATING_POINT_ACCURACY,
                        CALC_TBS,
                        TEXTURE_PROCESSOR,
                        BAKE_LAYERS)

    #bpy.ops.wm.addon_enable(module='io_scene_egg')
    #bpy.ops.export.panda3d_egg(pth)


def transformModel():
    
    ## Select the meshes
    bpy.ops.object.select_by_type(type='MESH')

    ## Center model
    #not sure how, not necessary at the moment
    # bpy.ops.view3d.snap_cursor_to_center()
    # bpy.ops.view3d.snap_selected_to_cursor()

    ## Scale
    Dim = [0., 0., 0]
    for obj in bpy.context.selected_objects:
        for i, d in enumerate(obj.dimensions):
            Dim[i] = max((Dim[i], d))
    scale = [1. / max(Dim)] * 3
    bpy.ops.transform.resize(value=scale)

    # ## Rotate
    # rotx = (270.,)
    # rotz = (180.,)
    # bpy.ops.transform.rotate(value=rotx, axis=(1., 0., 0.))
    # bpy.ops.transform.rotate(value=rotz, axis=(0., 0., 1.))


def doStuff():
    """
    Custom code
    """

    transformModel()


def run(pth):
    # Empty the current scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    
    # Import obj into scene
    importObj(pth)

    # Do whatever you need to do within the Blender scene
    doStuff()

    # Export egg
    outpth = os.path.splitext(pth)[0] + '.egg'
    exportEgg(outpth)

    # Drop to debugger
    print("\nYou're now in the debugger, within the Blender context\n")
    pdb.set_trace()



# Main
if __name__ == "__main__":
    # Get command line arguments
    args = sys.argv
    
    # Get the relevant arguments, which follow the "--" sign
    pyargs = args[args.index("--") + 1:]

    # Get the indicated filename
    pth = pyargs[0]

    # Run the import
    run(pth)

