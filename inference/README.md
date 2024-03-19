# Deep Object Pose Estimation (DOPE) - Inference 

This directory contains a simple example of inference for DOPE.


## Setup

If you haven't already, install the dependencies listed in `requirements.txt`
in the root of the repo:

```
pip install -r ../requirements.txt
```

## Running Inference

The `inference.py` script will take a trained model to run inference. In order to run, the following 3 arguments are needed: 
1. `--weights`: path to the trained model weights. Can either point to a single `.pth` file or a folder containing multiple `.pth` files. If this path points to a folder with multiple `.pth` files, the script will individually load and run inference for all of the weights.
2. ``--data`: path to the data that will be used as input to run inference on. The script **recursively** loads all data that end with extensions specified in the `--exts` flag.
3. `--object`: name of the class to run detections on. This name must be defined under `dimensions` in the config file passed to `--config`.

Below is an example of running inference:

```
python inference.py --weights ../weights --data ../sample_data --object cracker
```

### Configuration Files
Depending on the images you want to run inference on, you may need to redefine the configuration values in `camera_info.yaml` and `config_pose.yaml`.
You can either define a new configuration file and specify it with `--config` and `--camera` or update `camera_info.yaml` and `config_pose.yaml`.

Before running inference, it is important to make sure that: 
1. The `projection_matrix` field is set properly in `camera_info.yaml` (or the file you specified for `--camera`). 
The `projection_matrix` field should be a `3x4` matrix of the form:
```
[fx,   0,  cx,  0,
  0,  fy,  cy,  0,
  0,   0,   1,  0]
```

2. The `dimensions` and `class_ids` fields have been specified for the object you wish to detect in `config_pose.yaml` (or the file you specified for `--config`).

### Running Inference with Multiple Weights at Once
The inference script can run inference on multiple weights if the path specified in ``--weights`` points to a folder containing multiple `.pth` files.
This feature is useful for fast evaluation of multiple weights to find the epoch that performs the best.
While, generally, later epochs tend to perform better than earlier ones, this is not always the case.
For more information on how to quantitatively evaluate the performance of a trained model, refer to the `/evaluate` subdirectory.  
