# Deep Object Pose Estimation (DOPE) - Training 

This repo contains a simplified version of the **training** script for DOPE.
The original repo for DOPE [can be found here](https://github.com/NVlabs/Deep_Object_Pose). 

In addition, this repo contains scripts for inference, evaluation, and data visualization.
More instructions can be found in the subdirectories `/evaluate` and `/inference`.

## Installing Dependencies
***Note***

It is highly recommended to install these dependencies in a virtual environment. You can create and activate a virtual environment by running: 
```
python -m venv ./output/dope_training
source ./output/dope_training/bin/activate
```
---
To install the required dependencies, run:
```
pip install -r requirements.txt
```

## Training
We use `torchrun` to run the training script.
At minimum, the ``--data`` and ``--object`` flags must be specified if training with data that is stored locally:
```
python -m torch.distributed.launch --nproc_per_node=1 train.py --data PATH_TO_DATA --object CLASS_OF_OBJECT
```
The ``--data`` flag specifies the path to the training data. There can be multiple paths that are passed in. 

The ``--object`` flag specifies the name of the object to train the DOPE model on.
Although multiple objects can be passed in, DOPE is designed to be trained for a specific object. For best results, only specify one object.
The name of this object must match the `"class"` field in groundtruth `.json` files.

To get a full list of the command line arguments, run `python train.py --help`.

### Loading Data from `s3`
There is also an option to train with data that is stored on an `s3` bucket. The script uses `boto3` to load data from `s3`.
The easiest way to configure credentials with `boto3` is with a config file, which you can [setup using this guide](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#aws-config-file).

When training with data from `s3`, be sure to specify the ``--use_s3`` flag and also the ``--train_buckets`` flag that indicates which buckets to use for training.
Note that multiple buckets can be specified with the `--train_buckets` flag. 

In addition, the `--endpoint` must be specified in order to retrieve data from an `s3` bucket. 

Below is a sample command to run the training script while using data from `s3`.
```
torchrun --nproc_per_node=1 train.py --use_s3 --train_buckets BUCKET_1 BUCKET_2 --endpoint ENDPOINT_URL --object CLASS_OF_OBJECT
```

### Multi-GPU Training

To run on multi-GPU machines, set `--nproc_per_node=<NUM_GPUs>`. In addition, reduce the number of epochs by a factor of the number of GPUs you have.
For example, when running on an 8-GPU machine, setting ``--epochs 5`` is equivalent to running `40` epochs on a single GPU machine.

## Debugging 
There is an option to visualize the `projected_cuboid_points` in the ground truth file. To do so, run:
```
python debug.py --data PATH_TO_IMAGES
```

## Common Issues

1. If you notice you are running out of memory when training, reduce the batch size by specifying a smaller ``--batchsize`` value. By default, this value is `32`.
2. If you are running into dependency issues when installing, 
you can try to install the version specific dependencies that are commented out in `requirements.txt`. Be sure to do this in a virtual environment.

## License

Copyright (C) 2018 NVIDIA Corporation. All rights reserved. Licensed under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
