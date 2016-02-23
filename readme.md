# Introduction

Genthor is about integrating generative models and Bayesian techniques with thor.


# Genthor setup

- Clone and install `genthor`:

```
cd PATH_TO_GENTHOR
git clone https://github.com/dicarlolab/genthor.git
cd genthor
python setup.py install
```

- Copy `obj_Bscript.py` to your `~/.skdata/genthor` path:

```
mkdir ~/.skdata/genthor
cp PATH_TO_GENTHOR/genthor/genthor/modeltools/obj_Bscript.py ~/.skdata/genthor
```

- Add genthor dependencies to your Python path. On `mh17`, they are already available at `/usr/local/lib/python2.7/site-packages`. Otherwise, read about [installing dependencies](#installing-dependencies)


# Installing dependencies

## General packages

(You must have root to install these packages)

```
apt-get install bison flex
apt-get install -y -q libgl1-mesa-dev
apt-get install -y -q libtiff4-dev
apt-get install -y -q libsdl1.2-dev
```

## Blender

- Download (get the link to the correct version and operating system [on the Blender website](http://www.blender.org/download/get-blender/):

```
wget http://download.blender.org/release/Blender2.68/blender-2.68a-linux-glibc211-x86_64.tar.bz2
```

- Extract: `tar jxf blender-2.68a-linux-glibc211-x86_64.tar.bz2`

- Copy the modified folders `io_scene_egg` and `io_scene_obj` from the genthor install to Blender's addons:

  - If using a local installation
  ```
  cp -r PATH_TO_GENTHOR/genthor/io_scene_egg PATH_TO_BLENDER_INSTALL/2.68/scripts/addons/
  cp -r PATH_TO_GENTHOR/genthor/io_scene_obj PATH_TO_BLENDER_INSTALL/2.68/scripts/addons/
  # For example:
  # cp -r ~/genthor/io_scene_obj ~/render/blender-2.68a-linux-glibc211-x86_64/2.68/scripts/addons/ 
  
  ```

  - If using a system-wide installation (check Blender version using `/usr/local/bin/blender --version`):
  ```
  cp -r PATH_TO_GENTHOR/genthor/io_scene_egg ~/.config/blender/2.68/scripts/addons/
  cp -r PATH_TO_GENTHOR/genthor/io_scene_obj ~/.config/blender/2.68/scripts/addons/
  ```

  *NOTE:* If you previously have used your local installation of blender and wish to use the global one, you need to delete `~/bin/blender`.Otherwise, your local version will be continued to be used.

- (Local installation only) Link `/bin/blender` to your install location:

```
ln -s ~/PATH_TO_BLENDER_INSTALL/blender ~/bin/blender
# For example:
# ln -s ~/render/blender-2.68a-linux-glibc211-x86_64/blender ~/bin/blender
```

## Panda

- Download panda from source: `wget https://www.panda3d.org/download/panda3d-1.8.1/panda3d-1.8.1.tar.gz`

- Unzip: `tar -zxvf panda3d-1.8.1.tar.gz`

- Make: `python makepanda/makepanda.py --everything --no-vision` (make sure you are in the folder where you installed/extracted panda before you run makepanda)

- Edit panda's `Config.prc` file (locate or find Config.prc, nano or vi the file) so it looks like this:

```
#load-display pandagl
#load-display pandadx9
#load-display pandadx8
#load-display pandagles
load-display p3tinydisplay
```

*NOTE:* The point of making this change is to allow panda to render to an off-screen buffer, so that it can run on machines without displays (e.g. servers). This is an installation-specific thing that, if you were running on your own personal desktop and wanted to see things on the screen as they rendered, you could leave it as it was (e.g. `pandagl`).

- You can then install it in the same location (path to install has to be the same as path you called `makepanda`):

```
python makepanda/installpanda.py --prefix=/[PATHYOUWANTTOINSTALLIN]
```

- Add `[PATH_TO_PANDA_INSTALL]/usr/local/share/panda3d` and `[PATH_TO_PANDA_INSTALL]/usr/local/share/panda3d` to your Python path in your `.profile` so this is done automatically at login:

```
export PYTHONPATH=$PYTHONPATH:[PATH_TO_PANDA_INSTALL]/lib/panda3d:[PATH_TO_PANDA_INSTALL]/share/panda3d
```


# Bare minimum testing (in Python)

```python
import genthor.datasets as gd
# When you run genthor for the first time, it will need to load background and models. 
# You can do this by:
meta = dataset.meta
# after define your models, you will need to load models
dataset.get_models()
```


# Converting `.obj` to `.egg`

To convert .obj to .egg, you need Blender and the Yabee exporter add-on. The code was tested on Blender 2.63 and Yabee r12 for Blender2.63a. See blender.org and code.google.com/p/yabee for those versions.


# Resource organization

Here is the organization for your models and backgrounds that you must observe if you want model/background loading to work properly.

The `resources` directory is the root of all models and backgrounds, and within `resources` there are up to 5 sub-directories:

- `resources/backgrounds` stores images for backgrounds.

- `objs` / `eggs` / `bams` stores mesh models, each within a subdirectory with the model's name, e.g.: `resources/eggs/bloodhound/bloodhound.egg`. Within each model directory, there is a sub-directory called `tex` that stores texture images.  The `objs` directory can contain `.tgz` or other tar-zip readable files instead.

- `raw_models` contains `.obj`/ `.tgz` files that you can use `convert_models.py` to add to the official model directories above.
