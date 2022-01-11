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
python single_video_pybullet.py --nb_frames 1
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
