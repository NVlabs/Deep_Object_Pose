## DOPE in a Docker Container

Running ROS inside of [Docker](https://www.docker.com/) is an excellent way to
experiment with DOPE, as it allows the user to completely isolate all software and configuration
changes from the host system.  This document describes how to create and run a
Docker image that contains a complete ROS environment that supports DOPE,
including all required components, such as ROS Noetic, rviz, CUDA with cuDNN,
and other packages.

The current configuration assumes all components are installed on an x86 host
platform running Ubuntu 18.04 or later.  Further, use of the DOPE Docker container requires an NVIDIA GPU to be present, and the use of Docker version 19.03.0 or later.


### Steps

1. **Download the DOPE code**
   ```
   $ git clone https://github.com/NVlabs/Deep_Object_Pose.git dope
   ```

2. **Build the docker image**
   ```
   $ cd dope/docker
   $ docker build -t nvidia-dope:noetic-v1 -f Dockerfile.noetic ..
   ```
   This will take several minutes and requires an internet connection.

3. **Plug in your camera**
   Docker will not recognize a USB device that is plugged in after the container is started.

4. **Run the container**
   ```
   $ ./run_dope_docker.sh [name] [host dir] [container dir]
   ```
   Parameters:
   - `name` is an optional field that specifies the name of this image. By default, it is `nvidia-dope-v2`.  By using different names, you can create multiple containers from the same image.
   - `host dir` and `container dir` are a pair of optional fields that allow you to specify a mapping between a directory on your host machine and a location inside the container.  This is useful for sharing code and data between the two systems.  By default, it maps the directory containing dope to `/root/catkin_ws/src/dope` in the container.

      Only the first invocation of this script with a given name will create a container. Subsequent executions will attach to the running container allowing you -- in effect -- to have multiple terminal sessions into a single container.

5. **Build DOPE**
   Return to step 7 of the [installation instructions](../readme.md) (downloading the weights).

   *Note:* Since the Docker container binds directly to the host's network, it will see `roscore` even if running outside the docker container.
