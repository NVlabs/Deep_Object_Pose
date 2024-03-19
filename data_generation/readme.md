# Synthetic Data Generation

This directory contains code for data generation (both
images and associated JSON files) for training DOPE.  We provide
two variations of this code, one that uses [NVISII](https://github.com/owl-project/NVISII) for rendering, and another that
uses [Blenderproc](https://github.com/DLR-RM/BlenderProc).
The data produced by these pipelines are compatible with each
other, and can be combined into a single dataset without
issues.

We highly recommend a GPU with 
[RTX capabilities](https://www.nvidia.com/en-us/geforce/news/geforce-gtx-dxr-ray-tracing-available-now/), as ray tracing can be costly on a non-RTX GPU. 


## Setup

### Environment Maps
You will need to download HDRI maps to illuminate the scene. These can be found freely on [polyhaven](https://polyhaven.com/hdris). 
For testing purposes, you can download a single one here: 
```
wget https://www.dropbox.com/s/na3vo8rca7feoiq/teatro_massimo_2k.hdr
mv teatro_massimo_2k.hdr dome_hdri_haven/
```


### Distractors

In addition to your object of interest, we recommend adding "distractor" objects to the scene. These are unrelated objects that serve to enrich the training data, provide occlusion, etc.
Our data generation scripts use the [Google scanned objects dataset](https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects), which can be downloaded automatically with the following: 

```
python download_google_scanned_objects.py
```


# Dataset format

This section contains a description of the generated dataset format.

## Coordinate systems and units

All coordinate systems (world, model, camera) are right-handed.

The world coordinate system is X forward, Y left and Z up.

The model coordinate system is ultimately defined by the mesh, but if the model should appear "naturally upright" in its neutral orientation in the world frame, the Z axis should point up (when the object is standing "naturally upright"), the X axis should point from the "natural backside" of the model towards the front, the Y axis should point left and the origin should coincide with the center of the 3D bounding box of the object model.

The camera coordinate system is the same as in OpenCV with X right, Y down and Z into the image away from the viewer.

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
      |         ^ z     |  |
      |         |       |  |
      |    y <--x       |  |
      |  |              |  |
   (y) 7 +--            |  + 4 (g)
      | /               | /
      |/                |/
(y) 6 +-----------------+ 5 (g)
```
Debug markers for the cuboid corners can be rendered using the `--debug` option, with (b) = blue, (m) = magenta, (g) = green, (y) = yellow and the centroid being white.

## JSON Fields

Each generated image is accompanied by a JSON file. This JSON file **must** contain the following fields to be used for training:

* `objects`: An array, containing one entry for each object instance, with:
    - `class`: class name. This name is referred to in configuration files.
    - `location` and `quaternion_xyzw`: position and orientation of the object in the *camera* coordinate system
    - `projected_cuboid`: 2D coordinates of the projection of the the vertices of the 3D bounding cuboid (in pixels) plus the centroid. See the above section "Projected Cuboid Corners" for more detail.
    - `visibility`: The visible fraction of the object silhouette (= `px_count_visib`/`px_count_all`). 
      Note that if NVISII is used, the object may still not be fully visible when `visibility == 1.0` because it may extend beyond the borders of the image.
      

### Optional Fields
These fields are not required for training, but are used for debugging and numerical evaluation of the results.  We recommend generating this data if possible.

* `camera_data`
    - `camera_view_matrix`: 4×4 transformation matrix from the world to the camera coordinate system.
    - `height` and `width`: dimensions of the image in pixels
    - `intrinsics`: the camera intrinsics


* `objects`
    - `local_cuboid`: 3D coordinates of the vertices of the 3D bounding cuboid (in meters); currently always `null`
    - `local_to_world_matrix`: 4×4 transformation matrix from the object to the world coordinate system
    - `name`: a unique string that identifies the object instance internally


<br><br>


# Handling objects with symmetries

If your object has any rotational symmetries, they have to be handled specially.

## Cylinder object

Here is a video that demonstrates what happens with a rotationally symmetric object if you do not specifiy the symmetries:

https://user-images.githubusercontent.com/320188/159683931-8e87f778-8711-4e54-9ad8-536cf5862e01.mp4

As you can see on the left side of that video, the cuboid corners (visualized as small colored spheres) rotate with the object. Because the object has a rotational symmetry, this results in two frames that are pixel-wise identical to have different cuboid corners. Since the cuboid corners are what DOPE is trained on, this will cause the training to fail.

The right side of the video shows the same object with a debug texture to demonstrate the "real" pose of the object. If your real object actually has a texture like this, it **does not** have any rotational symmetries in our sense, because two images where the cuboid corners are in different places will also not be pixel-wise identical due to the texture. Also, you only need to deal with rotational symmetries, not mirror symmetries for the same reason.

To handle symmetries, you need to add a `model_info.json` file (see the `models_with_symmetries` folder for examples). Here is the `model_info.json` file for the cylinder object:

```json
{
  "symmetries_discrete": [[1,  0,  0,  0,
                           0, -1,  0,  0,
                           0,  0, -1,  0,
                           0,  0, 0,   1]],
  "symmetries_continuous": [{"axis": [0, 0, 1], "offset": [0, 0, 0]}],
  "align_axes": [{"object": [0, 1, 0], "camera": [0, 0, 1]}, {"object": [0, 0, 1], "camera": [0, 1, 0]}]
}
```

As you can see, we have specified one *discrete* symmetry (rotating the object by 180° around the x axis) and one *continuous* symmetry (rotating around the z axis). Also, we have to specify how to align the axes. With the `align_axes` specified as above, the algorithm will:

1. Discretize `symmetries_continuous` into 64 discrete rotations.
2. Combine all discrete and continuous symmetries into one set of complete symmetry transformations.
3. Find the combined symmetry transformation such that when the object is rotated by that transformation,
    - the y axis of the object (`"object": [0, 1, 0]`) has the best alignment (smallest angle) with the z axis of the camera (`"camera": [0, 0, 1]`)
    - if there are multiple equally good such transformations, it will choose the obje where the z axis of the object (`"object": [0, 0, 1]`) has the best alignment with the y axis of the camera (`"camera": [0, 1, 0]`).

See below for a documentation of the object and camera coordinate systems.

With this `model_info.json` file, the result is the following:

https://user-images.githubusercontent.com/320188/159683953-0fe390ab-1d26-4395-ae15-352d360f3cd9.mp4

## Hex screw object

As another example, here's a rather unusual object that has a 60° rotational symmetry around the z axis. The `model_info.json` file looks like this:

```json
{
  "symmetries_discrete": [[ 0.5,   -0.866, 0,     0,
                            0.866,  0.5,   0,     0,
                            0,      0,     1,     0,
                            0,      0,     0,     1],
                          [-0.5,   -0.866, 0,     0,
                            0.866, -0.5,   0,     0,
                            0,      0,     1,     0,
                            0,      0,     0,     1],
                          [-1,      0,     0,     0,
                            0,     -1,     0,     0,
                            0,      0,     1,     0,
                            0,      0,     0,     1],
                          [-0.5,    0.866, 0,     0,
                           -0.866, -0.5,   0,     0,
                            0,      0,     1,     0,
                            0,      0,     0,     1],
                          [ 0.5,    0.866, 0,     0,
                           -0.866,  0.5,   0,     0,
                            0,      0,     1,     0,
                            0,      0,     0,     1]],
  "align_axes": [{"object": [0, 1, 0], "camera": [0, 0, 1]}]
}
```

The transformation matrices have been computed like this:

```python
from math import sin, cos, pi
for yaw_degree in [60, 120, 180, 240, 300]:
    yaw = yaw_degree / 180 * pi
    print([cos(yaw), -sin(yaw), 0, 0, sin(yaw), cos(yaw), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
```

The resulting symmetry-corrected output looks like this:

https://user-images.githubusercontent.com/320188/159683969-33a46225-94c0-43d8-b888-5e702ae3c31a.mp4

## Final remarks on symmetries

This symmetry handling scheme allows the data generation script to compute consistent cuboid corners for most rotations of the object. Note however that there are object rotations where the cuboid corners become unstable and "flip over" to a different symmetry transformation. For the cylinder object, this is when the camera looks at the top or bottom of the cylinder (not shown in the video above). For the hex screw object, this is also when the camera looks at the top or bottom or when the rotation is close to the 60° boundary between two transformations (this can be seen in the video). Rotations within a few degrees of the "flipping over" rotation will not be handled well by the trained network. Unfortunately, this cannot be easily avoided.

Further note that specifying symmetries also improves the recognition results for "almost-symmetrical" objects, where there are only minor non-symmetrical parts, such as most of the objects from the [T-LESS dataset](https://bop.felk.cvut.cz/datasets/).


