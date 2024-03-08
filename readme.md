[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)
# Deep Object Pose Estimation - ROS Inference

This is the official DOPE ROS package for detection and 6-DoF pose estimation of **known objects** from an RGB camera.  The network has been trained on the following YCB objects:  cracker box, sugar box, tomato soup can, mustard bottle, potted meat can, and gelatin box.  For more details, see our [CoRL 2018 paper](https://arxiv.org/abs/1809.10790) and [video](https://youtu.be/yVGViBqWtBI).


![DOPE Objects](dope_objects.png)

## Updates

2024/03/07 - New training code. New synthetic data generation code, using Blenderproc. Repo reorganization

2022/07/13 - Added a script with a simple example for computing the ADD and ADD-S metric on data. Please refer to [script/metrics/](https://github.com/NVlabs/Deep_Object_Pose/tree/master/scripts/metrics). 

2022/03/30 - Update on the NViSII script to handle [symmetrical objects](https://github.com/NVlabs/Deep_Object_Pose/tree/master/scripts/nvisii_data_gen#handling-objects-with-symmetries).  Also the NViSII script is compatible with the original training script. Thanks to Martin Günther. 

2021/12/13 - Added a NViSII script to generate synthetic data for training DOPE. See this [readme](https://github.com/NVlabs/Deep_Object_Pose/tree/master/scripts/nvisii_data_gen) for more details. We also added the update training and inference (without ROS) scripts for the NViSII paper [here](https://github.com/NVlabs/Deep_Object_Pose/tree/master/scripts/train2). 

2021/10/20 - Added ROS2 Foxy inference support through [Isaac ROS DOPE package](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation) for Jetson and x86+CUDA-capable GPU.

2021/08/07 - Added publishing belief maps. Thank you to Martin Günther. 

2020/03/09 - Added HOPE [weights to google drive](https://drive.google.com/open?id=1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg), [the 3d models](https://drive.google.com/drive/folders/1jiJS9KgcYAkfb8KJPp5MRlB0P11BStft), and the objects dimensions to config. [Tremblay et al., IROS 2020](https://arxiv.org/abs/2008.11822).  The HOPE dataset can be found [here](https://github.com/swtyree/hope-dataset/) and is also part of the [BOP challenge](https://bop.felk.cvut.cz/datasets/#HOPE)


<br>
<br>

## Tested Configurations

We have tested on Ubuntu 20.04 with ROS Noetic with an NVIDIA Titan X and RTX 2080ti with Python 3.8. The code may work on other systems.

---
***NOTE***

For hardware-accelerated ROS2 inference support, please visit [Isaac ROS DOPE](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/tree/main/isaac_ros_dope) which has been tested with ROS2 Foxy on Jetson AGX Xavier/JetPack 4.6 and on x86/Ubuntu 20.04 with RTX3060i.

---
<br>
<br>

## Synthetic Data Generation
Code and instructions for generating synthetic training data are found in the `data_generation` directory. There are two options for the render engine: you can use [NVISII](https://github.com/owl-project/NVISII) or [Blenderproc](https://github.com/DLR-RM/BlenderProc)

## Training
Code and instructions for training DOPE are found in the `train` directory.

## Inference
Code and instructions for command-line inference using PyTorch are found in the `inference` directory

## Evaluation
Code and instructions for evaluating the quality of your results are found in the `evaluate` directory

---

## YCB 3D Models

DOPE returns the poses of the objects in the camera coordinate frame.  DOPE uses the aligned YCB models, which can be obtained using [NVDU](https://github.com/NVIDIA/Dataset_Utilities) (see the `nvdu_ycb` command).

---

## HOPE 3D Models

![HOPE 3D models rendered in UE4](https://i.imgur.com/V6wX64p.png)

We introduce new toy 3d models that you download [here](https://drive.google.com/drive/folders/1jiJS9KgcYAkfb8KJPp5MRlB0P11BStft). 
The folders are arranged like the YCB 3d models organization. 
You can buy the real objects using the following links 
[set 1](https://www.amazon.com/gp/product/B071ZMT9S2), 
[set 2](https://www.amazon.com/gp/product/B007EA6PKS), 
[set 3](https://www.amazon.com/gp/product/B00H4SKSPS), 
and 
[set 4](https://www.amazon.com/gp/product/B072M2PGX9). 

The HOPE dataset can be found [here](https://github.com/swtyree/hope-dataset/) and is also part of the [BOP challenge](https://bop.felk.cvut.cz/datasets/#HOPE).

## How to cite DOPE 

If you use this tool in a research project, please cite as follows:
```
@inproceedings{tremblay2018corl:dope,
 author = {Jonathan Tremblay and Thang To and Balakumar Sundaralingam and Yu Xiang and Dieter Fox and Stan Birchfield},
 title = {Deep Object Pose Estimation for Semantic Robotic Grasping of Household Objects},
 booktitle = {Conference on Robot Learning (CoRL)},
 url = "https://arxiv.org/abs/1809.10790",
 year = 2018
}
```

## License

Copyright (C) 2018 NVIDIA Corporation. All rights reserved. Licensed under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).


## Acknowledgment

Thanks to Jeffrey Smith (jeffreys@nvidia.com) for creating the Docker image.


## Contact

Jonathan Tremblay (jtremblay@nvidia.com), Stan Birchfield (sbirchfield@nvidia.com)
