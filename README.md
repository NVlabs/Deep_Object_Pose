# Deep Object Pose Estimation (DOPE) - ROS2 Inference

This is a ROS2 package to make inference with DOPE ([original ROS1 repo](https://github.com/NVlabs/Deep_Object_Pose/tree/master)). The official code is adapted to work on the ROS2 framework with Python3.10 and the updated requirements. See the official repository for detailed info about DOPE.

## Installing

The code has been tested on Ubuntu 22.04 with ROS2 Humble and Python 3.10. 

1. **Clone the repository into your ros2 workspace**
    ```
    mkdir -p ~/ros2_ws/src
    cd ~/ros2_ws/src
    git clone https://github.com/Vanvitelli-Robotics/DOPE.git dope -b ros2_humble
    ```

2. **Install python dependencies**
    ```
    cd ~/ros2_ws/src/dope
    python3 -m pip install -r requirements.txt
    ```
    
3. **Build**
    ```
    cd ~/ros2_ws
    colcon build --packages-select dope
    ```
    
4. **Run the node**

   First, you need to configure your setup in */config/config_dope.yaml*, specifying the topics of your camera and the parameters of the target objects. Note that you have to also specifity the path of the weights resulting from the training for each object you want to inferred (also several objects simultaneously). You can load the weights in ```weights/``` folder (we don't provide any .pth due to the high dimension of the file).

   After configuration, you can start the inference running the node as follows:
    ```
    ros2 run dope_ros2 dope_node --ros-args --params-file ~/ros2_ws/src/dope/config/config_dope.yaml
    ```
   or, if you prefer, you can use the available launch file:
      ```
    ros2 launch dope_ros2 dope.launch.py
    ```
   

