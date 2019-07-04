## Running DOPE with a webcam

This tutorial explains how to:

1. start a ROS driver for a regular USB webcam
2. calibrate the camera **or** enter the camera intrinsics manually
3. rectify the images and publish them on a topic

Since DOPE relies solely on RGB images and the associated `camera_info` topic,
it is essential that the camera is properly calibrated to give good results.
Also, unless you are using a very low-distortion lens, the images should be
rectified before feeding them to DOPE.

### A. Starting a ROS driver for a USB webcam

In this tutorial, we're using the [usb_cam](http://wiki.ros.org/usb_cam)
ROS package. If this package is not working with your camera, simply google
around - nowadays there is a ROS driver for almost every camera.

1. Install the driver:

    ```bash
    sudo apt install ros-kinetic-usb-cam
    ```

2. Run the camera driver (enter each command in a separate terminal)

    ```bash
    roscore
    rosrun usb_cam usb_cam_node _camera_name:='usb_cam' _camera_frame_id:='usb_cam'
    ```

    See the [usb_cam wiki page](http://wiki.ros.org/usb_cam) for a list of all
    parameters.

3. Check that the camera is running:

    ```
    $ rostopic list
    [...]
    /usb_cam/camera_info
    /usb_cam/image_raw
    [...]
    $ rostopic hz /usb_cam/image_raw
    subscribed to [/usb_cam/image_raw]
    average rate: 30.001
     	min: 0.029s max: 0.038s std dev: 0.00280s window: 28
    ```

4. If you want, you can also run `rviz` to visualize the camera topic.

Since the camera is still uncalibrated, you should have seen the following
warning when starting the `usb_cam` node in step 2:

```
[ WARN] [1561548002.895791819]: Camera calibration file /home/******/.ros/camera_info/usb_cam.yaml not found.
```

Also, the camera_info topic is all zeros:

```bash
$ rostopic echo -n1 /usb_cam/camera_info
header:
  seq: 87
  stamp:
    secs: 1561548114
    nsecs: 388301085
  frame_id: "usb_cam"
height: 480
width: 640
distortion_model: ''
D: []
K: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
R: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
P: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
binning_x: 0
binning_y: 0
roi:
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: False
```

To fix this, we need to generate a file called
`~/.ros/camera_info/usb_cam.yaml` which holds the camera intrinsics. Either
follow step **B** or **C** to do this.

### B. Manually entering camera intrinsics

If you know the camera intrinsics of your webcam, you can simply generate a new
file `~/.ros/camera_info/usb_cam.yaml` which looks like this (the example is
for a Logitech C920 webcam with the following intrinsics: fx = 641.5,
fy = 641.5, cx = 320.0, cy = 240.0):


```
image_width: 640
image_height: 480
camera_name: usb_cam
camera_matrix:
  rows: 3
  cols: 3
  data: [641.5, 0, 320.0, 0, 641.5, 240.0, 0, 0, 1]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0, 0, 0, 0, 0]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]
projection_matrix:
  rows: 3
  cols: 4
  data: [641.5, 0, 320.0, 0, 0, 641.5, 240.0, 0, 0, 0, 1, 0]
```

After creating this file, restart the `usb_cam` driver for the changes to take
effect. The warning "Camera calibration file not found" should have
disappeared, and the `/usb_cam/camera_info` topic should reflect the values
entered above.

Since the camera intrinsics we supplied above do not specify distortion
coefficients, the image does not need to be rectified, so you can skip the
remaining steps and use the `/usb_cam/image_raw` topic as input for DOPE.

If you want to do proper calibration and rectification instead, skip step **B**
and continue with **C**.

### C. Calibrating the webcam

Follow the steps in [this tutorial](http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration).

In short, run these commands:

```bash
sudo apt install ros-kinetic-camera-calibration
rosrun camera_calibration cameracalibrator.py --size 6x7 --square 0.0495 image:=/usb_cam/image_raw camera:=/usb_cam   # adjust these values to your checkerboard
```

* Move your checkerboard around and make sure that you cover a good range of
  distance from the camera, all parts of the image, and horizontal and vertical
  skew of the checkerboard.
* When done, press "calibrate" and **wait** until the calibration is complete.
  This can take a long time (minutes or hours), depending on how many
  calibration samples you took. As long as the image window is frozen and
  `camera_calibration` hogs a CPU, it's still computing.
* Once the calibration has finished, the window will unfreeze. Press "save",
  then press "commit".

After this, the calibration info should have been saved to
`~/.ros/camera_info/usb_cam.yaml`. Restart the `usb_cam` driver for the changes
to take effect.


### D. Rectifying the images

1. Install `image_proc`:

    ```bash
    sudo apt install ros-kinetic-image-proc
    ```

2. Create a file called `usb_cam_image_proc.launch` with the following contents:

    ```xml
    <launch>
      <arg name="camera" default="usb_cam" />
      <arg name="num_worker_threads" default="4" />
      <arg name="manager" value="$(arg camera)_nodelet_manager" />

      <group ns="$(arg camera)">
        <node pkg="nodelet" type="nodelet" name="$(arg manager)" args="manager" output="screen">
           <param name="num_worker_threads" value="$(arg num_worker_threads)" />
        </node>

        <include file="$(find image_proc)/launch/image_proc.launch">
          <arg name="manager" value="$(arg manager)" />
        </include>
      </group>
    </launch>
    ```

3. Launch it:

    ```bash
    roslaunch usb_cam_image_proc.launch
    ```

This should publish the topic `/usb_cam/image_rect_color` (among others). You
can now use this topic as the input for DOPE.
