# VoxSeg for Visual Navigation - ROS Repository

Will Huey, Sean Brynjolfsson

Contains OVSeg repo (from: https://github.com/facebookresearch/ov-seg)

Contains rviz_lighting repo (from https://github.com/mogumbo/rviz_lighting.git)

## Installation 
- install rospkg to the python version you will be using. Ex:
```bash
conda activate voxseg
pip install rospkg
```
- follow the [ovseg installation instructions](https://github.com/facebookresearch/ov-seg/blob/main/INSTALL.md) to create a new python environment with the necessary dependencies for ovseg
- Download the largest ovseg model, [swinbase_vitL_14](https://github.com/facebookresearch/ov-seg/blob/main/GETTING_STARTED.md), and put it in src/modules/ovseg/models
- change the python shebang at the top of the files in voxseg/src/*.py to the python version with the necessary packages installed (for ovseg, isaacsim, wvn)
```python
#!/path/to/voxseg/python
```
- change the VOXSEG_ROOT_DIR environment variable in voxseg/modules/config.py, to match the root of the voxseg package
  - This will probably be .../catkin_ws/src/voxseg

- Build and source the workspace (run the following, from the workspace root):
```bash
catkin_make
source devel/setup.bash
```

## Instructions for running voxseg 
IMPORTANT: each new terminal you open to run the nodes, make sure you source devel/setup.bash
Basic Example:
- Modify the world_config and batch_size parameters in modules/config.py, according to your needs
- Launch the server, and wait for it to print "Backend Setup"
- WARNING: If you start sending messages while the models are loading, it might do strange things
```bash
roslaunch voxseg server.launch
```
- Edit data_dir and class_names in main of modules/send_classes.py, modules/send_images
- Publish image data + class names or prompts
```bash
roslaunch voxseg send_images.launch 
roslaunch voxseg send_classes.launch 
```
- Request clip inference on the image embeddings and class names 
```bash
roslaunch voxseg request_computation.launch
```
Command Line Options:
- Publish class names from command line:
```bash 
rostopic pub /'voxseg_classes_topic' voxseg/Classes "classes: [class_1, class_2]"
```
- Reset the server data from command line
```bash
rostopic pub /voxseg_reset_topic std_msgs/String “r”
```

## Instructions for running the voxel visualization in RViz:

- Source ros:
```bash
source /opt/ros/noetic/setup.bash
```
  
- Start roscore:

```bash
roscore
```

- get the markers by using get_ros_voxels 

- run rviz_voxel_node.py to start the voxel topic (by default, this is /voxel_grid_array)

Now, you have two options:

OPTION 1 - RVIZ WITH LIGHTING
- This is useful for seeing different colors and class names better, but not necessary
- Start rviz with rviz_lighting
```bash
roslaunch rviz_lighting rviz_lighting.launch
```
- To change the lighting, change the 'color' property under the AmbientLight in the Displays Window (white -> brighter, black -> darker)

OPTION 2 - DEFAULT RVIZ:
- Define the world frame (for a default frame, run the following)
```bash
rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map world 5
```
- Start rviz:
```bash
rosrun rviz rviz
```

Now, rviz should be running. To view the voxels:
- Go to displays > global options > fixed frame, and change it from map to world
  - if world doesn't appear, this step was probably skipped
- Go to "Add" (bottom left of RVIZ) > "By Topic", click on "MarkerArray" under "voxseg_marker_topic", and click "OK"

- If you would like to define your own markers, just publish them to voxseg_marker_topic. They will render with nice colors in rviz
- If you are using Voxseg, the voxels will be published as they are computed: 
  - The rviz communication node is listening on 'voxseg_voxels_topic'. This is automatically published to after computations are performed
  - In order for class names to show up, it is important that they are published while the rviz communication node is running. Otherwise, every class will be labeled "undefined"
- Limits: it starts getting pretty slow above 100x100x50 resolution (<1hz). It may be necessary to use a Marker.Cube_List for larger scenes

## Quick Setup
To roscore, catkin_make, and start the rviz nodes and the server node all in one go, run setup_terminals.bash from the root. Then, you just need to supply classes/images, and request a computation.

## Developer Information

Voxseg specific parameters can be found in modules/config.py

For text queries, you can choose to publish either class names or manually defined prompts. If you use class names, each of them will automatically be expanded into a preset list of about 20 prompt templates. If you use prompts, you must define each class id, and the corresponding prompts for it. 

All of the voxseg computations are handled by modules/server.py. The voxseg methods themself are in voxel_world.py.

The server is set up to listen on the following topics and services:

- IMAGE_TOPIC:
  - type: voxseg.DepthImageInfo.msg
  - contains rgb image, depths, and camera extrinsics
  - modules/client/VoxSegClient.publish_depth_image will calculate this message type from numpy arrays
  - The server adds data published here to a buffer of images to use for inference
- CLASS_TOPIC
  - type: voxseg.Classes.msg
  - contains a string list of class identifiers 
  - The server overwrites any previously stored class names with new names published here
- RESET_TOPIC
  - type: std_msgs.String.msg
  - resets all of the servers buffers, for a blank slate run
- VOXEL_REQUEST_SERVICE
  - type: std_msgs.int32
  - requests a voxel class computation from the server, and returns the voxels with their corresponding classes
  - requires that the server has defined class names and images

