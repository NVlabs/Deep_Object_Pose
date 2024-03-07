# Running DOPE with ROS

This directory and its subdirectories contains code for running DOPE with ROS Noetic.
The following steps assume you have installed ROS already.  Alternatively, you can use the provided [Docker image](docker/readme.md) and skip to Step #7.

1. **Install ROS**

    Follow these [instructions](http://wiki.ros.org/noetic/Installation/Ubuntu).
    You can select any of the default configurations in step 1.4; even the
    ROS-Base (Bare Bones) package (`ros-noetic-ros-base`) is enough.

2. **Create a catkin workspace** (if you do not already have one). To create a catkin workspace, follow these [instructions](http://wiki.ros.org/catkin/Tutorials/create_a_workspace):
    ```
    $ mkdir -p ~/catkin_ws/src   # Replace `catkin_ws` with the name of your workspace
    $ cd ~/catkin_ws/
    $ catkin_make
    ```

3. **Download the DOPE code**
    ```
    $ cd ~src
    $ git clone https://github.com/NVlabs/Deep_Object_Pose.git
    $ ln -s ~/src/Deep_Object_Pose/ros1/dope ~/catkin_ws/src/dope
    ```

4. **Install python dependencies**
    ```
    $ cd ~/catkin_ws/src/dope
    $ python3 -m pip install -r ~/src/Deep_Object_Pose/requirements.txt
    ```

5. **Install ROS dependencies**
    ```
    $ cd ~/catkin_ws
    $ rosdep install --from-paths src -i --rosdistro noetic
    $ sudo apt-get install ros-noetic-rosbash ros-noetic-ros-comm
    ```

6. **Build**
    ```
    $ cd ~/catkin_ws
    $ catkin_make
    ```

7. **Download [the weights](https://drive.google.com/open?id=1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg)** and save them to the `weights` folder, *i.e.*, `~/catkin_ws/src/dope/weights/`.


## Running

1. **Start ROS master**
    ```
    $ cd ~/catkin_ws
    $ source devel/setup.bash
    $ roscore
    ```

2. **Start camera node** (or start your own camera node)
    ```
    $ roslaunch dope camera.launch  # Publishes RGB images to `/dope/webcam_rgb_raw`
    ```

    The camera must publish a correct `camera_info` topic to enable DOPE to compute the correct poses. Basically all ROS drivers have a `camera_info_url` parameter where you can set the calibration info (but most ROS drivers include a reasonable default).

    For details on calibration and rectification of your camera see the [camera tutorial](doc/camera_tutorial.md).

3. **Edit config info** (if desired) in `~/catkin_ws/src/dope/config/config_pose.yaml`
    * `topic_camera`: RGB topic to listen to
    * `topic_camera_info`: camera info topic to listen to
    * `topic_publishing`: topic namespace for publishing
    * `input_is_rectified`: Whether the input images are rectified. It is strongly suggested to use a rectified input topic.
    * `downscale_height`: If the input image is larger than this, scale it down to this pixel height. Very large input images eat up all the GPU memory and slow down inference. Also, DOPE works best when the object size (in pixels) has appeared in the training data (which is downscaled to 400 px). For these reasons, downscaling large input images to something reasonable (e.g., 400-500 px) improves memory consumption, inference speed *and* recognition results.
    * `weights`: dictionary of object names and there weights path name, **comment out any line to disable detection/estimation of that object**
    * `dimensions`: dictionary of dimensions for the objects  (key values must match the `weights` names)
    * `class_ids`: dictionary of class ids to be used in the messages published on the `/dope/detected_objects` topic (key values must match the `weights` names)
    * `draw_colors`: dictionary of object colors (key values must match the `weights` names)
    * `model_transforms`: dictionary of transforms that are applied to the pose before publishing (key values must match the `weights` names)
    * `meshes`: dictionary of mesh filenames for visualization (key values must match the `weights` names)
    * `mesh_scales`: dictionary of scaling factors for the visualization meshes (key values must match the `weights` names)
    * `overlay_belief_images`: whether to overlay the input image on the belief images published on /dope/belief_[obj_name]
    * `thresh_angle`: undocumented
    * `thresh_map`: undocumented
    * `sigma`: undocumented
    * `thresh_points`: Thresholding the confidence for object detection; increase this value if you see too many false positives, reduce it if  objects are not detected.

4. **Start DOPE node**
    ```
    $ roslaunch dope dope.launch [config:=/path/to/my_config.yaml]  # Config file is optional; default is `config_pose.yaml`
    ```




## Debugging

* The following ROS topics are published (assuming `topic_publishing == 'dope'`):
    ```
    /dope/belief_[obj_name]    # belief maps of object
    /dope/dimension_[obj_name] # dimensions of object
    /dope/pose_[obj_name]      # timestamped pose of object
    /dope/rgb_points           # RGB images with detected cuboids overlaid
    /dope/detected_objects     # vision_msgs/Detection3DArray of all detected objects
    /dope/markers              # RViz visualization markers for all objects
    ```
    *Note:* `[obj_name]` is in {cracker, gelatin, meat, mustard, soup, sugar}

* To debug in RViz, run `rviz`, then add one or more of the following displays:
    * `Add > Image` to view the raw RGB image or the image with cuboids overlaid
    * `Add > Pose` to view the object coordinate frame in 3D.
    * `Add > MarkerArray` to view the cuboids, meshes etc. in 3D.
    * `Add > Camera` to view the RGB Image with the poses and markers from above.

    If you do not have a coordinate frame set up, you can run this static transformation: `rosrun tf2_ros static_transform_publisher 0 0 0 0.7071 0 0 -0.7071 world <camera_frame_id>`, where `<camera_frame_id>` is the `frame_id` of your input camera messages.  Make sure that in RViz's `Global Options`, the `Fixed Frame` is set to `world`. Alternatively, you can skip the `static_transform_publisher` step and directly set the `Fixed Frame` to your `<camera_frame_id>`.

* If `rosrun` does not find the package (`[rospack] Error: package 'dope' not found`), be sure that you called `source devel/setup.bash` as mentioned above.  To find the package, run `rospack find dope`.

