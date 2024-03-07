# Deep Object Pose Estimation (DOPE) - Evaluation 

This repo contains a simplified version of the **evaluation** script for DOPE.
The original repo for DOPE [can be found here](https://github.com/NVlabs/Deep_Object_Pose). 

## Running Evaluation

After running inference with trained model weights, you can measure the performance of the model.

Below is an example of running the evaluation script:
```
python evaluate.py --data_prediction ../inference/output --data ../sample_data 
```
## Arguments 
### `--data`:
Path to groundtruth data for the predictions that you want to evaluate. 

### `--data_prediction`:
Path to predictions that were outputted from running inference. To support the evaluation of multiple sets of weights at once, this path can point to a folder containing the **outputs of multiple inference results**. 

### `--models`: 
Path to 3D model files. 
These models are loaded before running evaluation and are rendered to compute the 3D error between the predicted results and ground truth. 
Point this argument at the root of the folder containing all of your different model files. Below is a sample folder structure of what the models folder should look like: 

```
/PATH_TO_MODELS_FOLDER
├── 002_master_chef_can
│   ├── 002_master_chef_can.xml
│   ├── points.xyz
│   ├── textured.mtl
│   ├── textured.obj
│   ├── textured_simple.obj
│   ├── textured_simple.obj.mtl
│   └── texture_map.png
└── 035_power_drill
    ├── 035_power_drill.xml
    ├── points.xyz
    ├── textured.mtl
    ├── textured.obj
    ├── textured_simple.obj
    ├── textured_simple.obj.mtl
    └── texture_map.png
```

If you trained DOPE on a new object and want to evaluate its performance, make sure to include the 3D model files in a folder that matches `"class_name"` in the ground truth `.json` file. 

Multiple models can be loaded at once as the script will recursively search for any 3D models in the folder specified in `--models`.

### `--adds`:
The average distance computed using the closest point distance between the predicted pose and the ground truth pose.
This takes a while to compute. If you are only looking for a fast approximation, use ``--cuboid``.

### `--cuboid`:
Computes average distance using the 8 cuboid points of the 3D models.
It is much faster than ``--adds`` but is only an approximation for the metric. It should be used for testing purposes.
