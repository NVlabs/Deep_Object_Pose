[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)
# Deep Object Pose Estimation

This is the official repository for NVIDIA's Deep Object Pose Estimation, which performs detection and 6-DoF pose estimation of **known objects** from an RGB camera.  For full details, see our [CoRL 2018 paper](https://arxiv.org/abs/1809.10790) and [video](https://youtu.be/yVGViBqWtBI).


![DOPE Objects](dope_objects.png)


## Contents

This repository contains complete code for [training](train), [inference](inference), numerical [evaluation](evaluate) of results, and synthetic [data generation](data_generation).  We also provide a [ROS1 Noetic package](ros1) that performs inference on images from a USB camera.

Hardware-accelerated ROS2 inference can be done with the external
[NVIDIA Isaac ROS DOPE](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation/tree/main/isaac_ros_dope) project.


The [original version](CoRL) of the code used for the CoRL paper is also included
for reference, but is no longer being maintianed.

## Walkthrough
We provide a [walkthrough](walkthrough.md) of the entire pipeline: generating data, training a model, and inference.

## Datasets

We have trained and tested DOPE with two publicaly available datasets: YCB, and HOPE. The trained weights can be [downloaded from Google Drive](https://drive.google.com/drive/folders/1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg).



### YCB 3D Models
YCB models can be downloaded from the [YCB website](http://www.ycbbenchmarks.com/), or by  using [NVDU](https://github.com/NVIDIA/Dataset_Utilities) (see the `nvdu_ycb` command).  


### HOPE 3D Models
The [HOPE dataset](https://github.com/swtyree/hope-dataset/) is a collection of RGBD images and video sequences with labeled 6-DoF poses for 28 toy grocery objects.  The 3D models [can be  downloaded here](https://drive.google.com/drive/folders/1jiJS9KgcYAkfb8KJPp5MRlB0P11BStft). 
The folders are organized in the style of the YCB 3d models. 

The physical objects can be purchased online (details and links to Amazon can be found in the [HOPE repository README](https://github.com/swtyree/hope-dataset/).



## Tested Configurations

We have tested our standalone training, inference and evaluation scripts on Ubuntu 20.04 and 22.04 with Python 3.8+, using an NVIDIA Titan X, 2080Ti, and Titan RTX. 

The ROS1 node has been tested with ROS Noetic using Python 3.10. The Isaac ROS2 DOPE node has been tested with ROS2 Foxy on Jetson AGX Xavier with JetPack 4.6; and on x86/Ubuntu 20.04 with a NVIDIA Titan X, 2080Ti, and Titan RTX.  

<br>
<br>

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

Copyright (C) 2018-2025 NVIDIA Corporation. All rights reserved. This code is licensed under the [NVIDIA Source Code License](license.md).


## Acknowledgment

Thanks to Jeff Smith (jeffreys@nvidia.com) for help maintaining the repo and software.  Thanks also to [Martin Günther](https://github.com/mintar) for his code contributions and fixes.  


## Contact

Jonathan Tremblay (jtremblay@nvidia.com), Stan Birchfield (sbirchfield@nvidia.com)
