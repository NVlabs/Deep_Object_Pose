[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)
# Deep Object Pose Estimation

This is the official repository for NVIDIA's Deep Object Pose Estimation, which performs detection and 6-DoF pose estimation of **known objects** from an RGB camera.  For full details, see our [CoRL 2018 paper](https://arxiv.org/abs/1809.10790) and [video](https://youtu.be/yVGViBqWtBI).


![DOPE Objects](dope_objects.png)


## Contents
This repository contains complete code for [training](train), [inference](inference), numerical [evaluation](evaluate) of results, and synthetic [data generation](data_generation) using either  [NVISII](https://github.com/owl-project/NVISII) or [Blenderproc](https://github.com/DLR-RM/BlenderProc).  We also provide a [ROS1 Noetic package](ros1) that performs inference on images from a USB camera.

Hardware-accelerated ROS2 inference can be done with the
[Isaac ROS DOPE](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/tree/main/isaac_ros_dope) project.


### A Note On Repo Organization
* `train2` contains the original training code, used to generate the results in the CORL paper. There have been some minor bug fixes, but this code will remain largely untouched in the future.
* Similarly, the synthetic data generation code in `data_generation/nvisii_data_gen/` was used for the paper, but depends on a rendering library that is no longer maintained.
* `train` contains new training code that is intended to be simpler and easier for users to understand and modify. This code will be maintained, and any new features will be added here.
* The synthetic data generation code in `data_generation/blenderproc` is a replacement for the nvisii code, using a different rendering engine that is still actively maintained.



## Tested Configurations

We have tested our standalone training, inference and evaluation scripts on Ubuntu 20.04 and 22.04 with Python 3.8+, using an NVIDIA Titan X, 2080Ti, and Titan RTX. 

The ROS1 node has been tested with ROS Noetic using Python 3.10. The Isaac ROS2 DOPE node has been tested with ROS2 Foxy on Jetson AGX Xavier with JetPack 4.6; and on x86/Ubuntu 20.04 with a NVIDIA Titan X, 2080Ti, and Titan RTX.  


## Datasets

We have trained and tested DOPE with two publicaly available datasets: YCB, and HOPE. The trained weights can be [downloaded from Google Drive](https://drive.google.com/drive/folders/1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg).



### YCB 3D Models
YCB models can be downloaded from the [YCB website](http://www.ycbbenchmarks.com/), or by  using [NVDU](https://github.com/NVIDIA/Dataset_Utilities) (see the `nvdu_ycb` command).  


### HOPE 3D Models
The [HOPE dataset](https://github.com/swtyree/hope-dataset/) is a collection of RGBD images and video sequences with labeled 6-DoF poses for 28 toy grocery objects.  The 3D models [can be  downloaded here](https://drive.google.com/drive/folders/1jiJS9KgcYAkfb8KJPp5MRlB0P11BStft). 
The folders are organized in the style of the YCB 3d models. 

The physical objects can be purchased online (details and links to Amazon can be found in the [HOPE repository README](https://github.com/swtyree/hope-dataset/).

<br><br>

---



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

Copyright (C) 2018-2024 NVIDIA Corporation. All rights reserved. This code is licensed under the [NVIDIA Source Code License](https://github.com/NVlabs/HANDAL/blob/main/LICENSE.txt).


## Acknowledgment

Thanks to Jeff Smith (jeffreys@nvidia.com) for help maintaining the repo and software.  Thanks also to [Martin Günther](https://github.com/mintar) for his code contributions and fixes.  


## Contact

Jonathan Tremblay (jtremblay@nvidia.com), Stan Birchfield (sbirchfield@nvidia.com)
