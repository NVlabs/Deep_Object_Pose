#!/bin/bash
#

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

# Stop in case of any error.
set -e

source /opt/ros/noetic/setup.bash

# Create catkin workspace.
mkdir -p ${CATKIN_WS}/src
cd ${CATKIN_WS}/src
catkin_init_workspace
# Clone ROS libraires that must be built from source
git clone https://github.com/ros-perception/camera_info_manager_py.git
cd ..
catkin_make
