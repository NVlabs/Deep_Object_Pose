## DOPE ROS2 in a Docker Container

You can run DOPE directly in a docker container with Ubuntu 22.04 with ROS2 Humble and Python 3.10.

### Steps

1. **Download the DOPE code**
   ```
   $ git clone https://github.com/Vanvitelli-Robotics/DOPE.git dope -b ros2_humble
   ```

2. **Build the docker image**
   ```
   $ cd dope/docker
   $ docker build -t dope-ros2:humble-v1 -f Dokerfile.humble ..
   ```
   This will take several minutes and requires an internet connection. If the build fails check if the dope/docker/init_workspace.sh has the execution rights.

3. **Plug in your camera**
   Docker will not recognize a USB device that is plugged in after the container is started.

4. **Run the container**
   ```
   $ ./run_docker.sh [name] [host dir] [container dir]
   ```
   Parameters:
   - `name` is an optional field that specifies the name of this image. By default, it is `dope-ros2-v2`.  By using different names, you can create multiple containers from the same image.
   - `host dir` and `container dir` are a pair of optional fields that allow you to specify a mapping between a directory on your host machine and a location inside the container.  This is useful for sharing code and data between the two systems.  By default, it maps the directory containing dope to `/root/ros2ws/src/dope` in the container.

      Only the first invocation of this script with a given name will create a container. Subsequent executions will attach to the running container allowing you -- in effect -- to have multiple terminal sessions into a single container.

5. **Build DOPE**
   Return to step 2 of the [installation instructions](../readme.md).

   *Note:* Since the Docker container binds directly to the host's network, it will see the `ros2 network` even if running outside the docker container.

   

