# UPDATES

- 11/01/2022: Added the possility to load a single object with `--path_single_obj`. Just give the direct path to the object. 
This function uses [nvisii.import_scene()](https://nvisii.com/nvisii.html#nvisii.import_scene). 
If the obj file is complex, it will break the object into sub components, 
so you might not have the projected cuboid, and you will get each pose of the different components with the cuboid. 
Be careful using this one, make sure your understand the implications. 
TODO: track the cuboid of the import_scene from nvisii.    


# Description

These sample scripts use [NViSII](https://github.com/owl-project/NVISII) to generate synthetic data for training the [DOPE](https://github.com/NVlabs/Deep_Object_Pose) object pose estimator. 
The data can also be used for training other networks.
To generate the data, you will need NVIDIA drivers 450 or above. 
We also highly recommend a GPU with RTX capabilities, as ray tracing can be costly on a non-RTX GPU. 

# Installation
```
pip install -r requirements.txt
```

## HDRI maps
You will need to download HDRI maps to illuminate the scene. These can be found freely on [polyhaven](https://polyhaven.com/hdris). 
For testing purposes, you can download a single one here: 
```
wget https://www.dropbox.com/s/na3vo8rca7feoiq/teatro_massimo_2k.hdr
mv teatro_massimo_2k.hdr dome_hdri_haven/
```


## Distractors

The script, as is, expects some objects to be used as distractors.  It is currently using the [Google scanned objects dataset](https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects), which can be download automatically with the following: 

```
python download_google_scanned_objects.py
```

If you do *not* want to use the distractors, use the following argument when running the script:  `--nb_distractors 0`.

# Running the script

If you downloaded everything from the previous steps, _e.g._, a single HDRI map and some distractors from Google scanned objects, you can run the following command:

```
python single_video_pybullet.py --nb_frames 1 --scale 0.01
```

This will generate a single frame example in `output/output_example/`. The image should be similar to the following: 

![This is an image](/scripts/nvisii_data_gen/output/output_example/00001.png)

The script has a few controls that are exposed at the beginning of the file. 
Please consult `single_video_pybullet.py` for a complete list of parameters. 
The major parameters are as follows: 
- `--spp` for the number of sample per pixel, the higher it is the better quality the resulting image.  
- `--nb_frames` number of images to export.
- `--outf` folder to store the data. 
- `--nb_objects` the number of objects to load, this can reload the same object multiple times. 
- `--nb_distractors` how many objects to add as distractors, this uses 3D models from Google scanned objects. 

# Adding your own 3D models 

You can simply use `--path_single_obj` to load your own 3d model. But there are some limitations for exporting the meta data if the obj is complex. Try to have it as a single obj, e.g., not multiple textures, similar to the provided one in the repo. 

## Modifying the code to load your object

The script loads 3d models that are expressed in the format that was introduced by YCB dataset. 
But it is fairly easy to change the script to load your own 3d model, [NViSII](https://github.com/owl-project/NVISII) allows you to load different format 
as well, not just `obj` files. In `single_video_pybullet.py` find the following code: 

```python
for i_obj in range(int(opt.nb_objects)):

    toy_to_load = google_content_folder[random.randint(0,len(google_content_folder)-1)]

    obj_to_load = toy_to_load + "/google_16k/textured.obj"
    texture_to_load = toy_to_load + "/google_16k/texture_map_flat.png"
    name = "hope_" + toy_to_load.split('/')[-2] + f"_{i_obj}"
    adding_mesh_object(name,obj_to_load,texture_to_load,scale=0.01)
```
You can change the `obj_to_load` and `texture_to_load` to match your data format. If your file format is quite different, for example you are using a `.glb` file, then in the function `adding_mesh_object()` you will need to change the following: 

```python
    if obj_to_load in mesh_loaded:
        toy_mesh = mesh_loaded[obj_to_load] 
    else:
        toy_mesh = visii.mesh.create_from_file(name,obj_to_load)
        mesh_loaded[obj_to_load] = toy_mesh
```
`visii.mesh.create_from_file` is the function that is used to load the data, this can load different file format. The rest of that function also loads the right texture as well as applying a material. The function also creates a collision mesh to make the object move. 

# Extra

This script is close to what was used to generate the data called `dome` in our NViSII [paper](https://arxiv.org/abs/2105.13962). 

If you use this data generation script in your research, please cite as follows: 

```latex
@misc{morrical2021nvisii,
      title={NViSII: A Scriptable Tool for Photorealistic Image Generation}, 
      author={Nathan Morrical and Jonathan Tremblay and Yunzhi Lin and Stephen Tyree and Stan Birchfield and Valerio Pascucci and Ingo Wald},
      year={2021},
      eprint={2105.13962},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
``` 

# Training

Please use the updated training scripts with this data: https://github.com/NVlabs/Deep_Object_Pose/tree/master/scripts/train2 . 

<!-- # To verify

- Verify that the data exported is compatible with the training script directly. This script does not export `_camera_setting.json` file for example, the information is directly in the `.json` files. 
PRs are welcome :P. 
 -->

# Dataset format

This section contains a description of the generated dataset format.

## Coordinate systems and units

All coordinate systems (world, model, camera) are right-handed.

The world coordinate system is X forward, Y left and Z up.

The model coordinate system is ultimately defined by the mesh, but if the model should appear "naturally upright" in its neutral orientation in the world frame, the Z axis should point up (when the object is standing "naturally upright"), the X axis should point from the "natural backside" of the model towards the front, the Y axis should point left and the origin should coincide with the center of the 3D bounding box of the object model.

The camera coordinate system is X right, Y up and Z out of the image towards the viewer.

All pixel coordinates (U, V) have the origin at the top left corner of the image, with U going right and V going down.

All length measurements are in meters.

## Projected cuboid corners

The indices of the 3D bounding cuboid are in the order shown in the sketch below (0..7), with the object being in its neutral orientation (X axis pointing forward, Y left, Z up).

The order of the indices is the same as NVidia Deep learning Dataset Synthesizer (NDDS) and nvdu_viz from NVidia Dataset Utilities.

```text
   (m) 3 +-----------------+ 0 (b)
        /                 /|
       /                 / |
(m) 2 +-----------------+ 1| (b)
      |                 |  |
      |       ^ z       |  |
      |       |         |  |
      |  y <--x         |  |
  (y) |                 |  + 4 (g)
      |                 | /
      |                 |/
(y) 6 +-----------------+ 5 (g)
```
Debug markers for the cuboid corners can be rendered using the `--debug` option, with (b) = blue, (m) = magenta, (g) = green, (y) = yellow and the centroid being white.

## JSON Fields

Each generated image is accompanied by a JSON file. This JSON file contains the following fields:

* `camera_data`
    - `camera_look_at`: an alternative representation of the `camera_view_matrix`
    - `camera_view_matrix`: 4×4 transformation matrix from the world to the camera coordinate system
    - `height` and `width`: dimensions of the image in pixels
    - `intrinsics`: the camera intrinsics
    - `location_world` and `quaternion_world_xyzw`: see below

* `objects`: one entry for each object instance, with:
    - `bounding_box_minx_maxx_miny_maxy`: 2D bounding box of the object in the image: left, right, top, bottom (in pixels)
    - `class`: class name
    - `local_cuboid`: 3D coordinates of the vertices of the 3D bounding cuboid (in meters); currently always `null`
    - `local_to_world_matrix`: 4×4 transformation matrix from the object to the world coordinate system
    - `location` and `quaternion_xyzw`: position and orientation of the object in the *camera* coordinate system
    - `location_world` and `quaternion_xyzw_world`:  position and orientation of the object (or camera) in the *world* coordinate system
    - `name`: unique string that identifies the object instance internally
    - `projected_cuboid`: 2D coordinates of the projection of the the vertices of the 3D bounding cuboid (in pixels) plus the centroid. See section "Projected cuboid corners".
    - `provenance`: always `nvisii`
    - `segmentation_id`: segmentation instance ID; unique integer value that is used for this object instance in the `.seg.exr` file
    - `visibility`: 1 if at least one pixel of the object is visible in the image, else 0
