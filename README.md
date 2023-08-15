# Open Vocabulary Traversability (OVT)

Compute CLIP embeddings live on a jetson, construct a voxel representation of the embeddings, and visualize them through open vocabulary scene queries. 

This repo provides full support for rviz visualization. For visualizations in Nvidia Isaac Sim, see https://github.com/jolfss/voxvis

![alternative](/src/voxseg/src/good/voxseg_merged.png)
Red voxels are classified as obstacles, yellow as untraversable, and green as traversable.

![alternative](/src/voxseg/src/good/merged_good.png)
Blue represents the mask generated for traversable terrain

## Installation 
- Create a new conda environment, and install
```bash
conda create -n voxseg
conda activate voxseg
```
- follow the [ovseg installation instructions](https://github.com/facebookresearch/ov-seg/blob/main/INSTALL.md) to install necessary dependencies for ovseg in the voxseg environment
- Download the largest ovseg model, [swinbase_vitL_14](https://github.com/facebookresearch/ov-seg/blob/main/GETTING_STARTED.md), and put it in src/modules/ovseg/models
- Install dependencies
```bash
pip install rospkg
pip install overloading
```
- Install [liegroups](https://github.com/utiasSTARS/liegroups), we have found it is necessary to build it from source.
- enter the liegroups directory (where setup.py is) and call 
```bash
pip install .
```

- Build and source the workspace. To do so, run the following, from the workspace root. Note that the workspace root is the root of this repo, unless you installed voxseg to your own separate catkin workspace:
```bash
catkin build voxseg 
source devel/setup.bash
```

## Instructions for running open vocabulary segmentation on an Anymal D

- Start ros
```bash
roscore
```
- To run with the simulation
```bash
cd ~/catkin_ws
catkin build anymal_dodo_rsl
source devel/setup.bash
rosrun anymal_dodo_rsl sim.py
```
- To run with a rosbag
```bash
rosparam set use_sim_time true
rosbag play --clock *.bag
```
- Launch the elevation mapping node: (first install elevation mapping cupy)
```bash
cd ~/catkin_ws
catkin build elevation_mapping_cupy
source devel/setup.bash
roslaunch elevation_mapping_cupy anymal.launch
```
- Run the OVT predictor node
```bash
catkin build voxseg
source devel/setup.bash
conda activate ovseg
roslaunch voxseg ovt.launch 
```
- Send new classes and toggle image saving
```bash
catkin build voxseg
rostopic pub /ovt/classes_topic voxseg/Classes "classes: [floor, other]"
rosparam set /ovt/SAVE_IMAGES false
```

## Instructions for running voxel segmentation
IMPORTANT: whenever you open a new terminal to run the nodes, make sure you run ```source devel/setup.bash```

Basic Example:
- Modify the world_config and batch_size parameters in modules/config.py, according to your needs
- Launch the server, and wait for it to print "Backend Setup"
- WARNING: If you start sending messages while the models are loading, it might do strange things
```bash
roslaunch voxseg server.launch
```
- Edit data_dir and class_names in main of modules/send_images.py, modules/send_classes.py
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
- Limits: rviz starts getting pretty slow above 100x100x50 voxel grid resolution (<1hz). For larger scenes, check out the Isaac Sim visualization repo, linked at the top of this file. Otherwise, publishing a Marker.Cube_List instead of an array of Marker.Cube objects may improve performance.

## Quick Setup
To roscore, catkin_make, and start the rviz nodes and the server node all in one go, run setup_terminals.bash from the root. Then, you just need to supply classes/images, and request a computation.

## Developer Information

Voxseg specific parameters can be found in modules/config.py

For text queries, you can choose to publish either class names or manually defined prompts. If you use class names, each of them will automatically be expanded into a preset list of about 20 prompt templates. If you use prompts, you must define each class id, and the corresponding prompts for it. 

You can also choose to combine class ids under groups. If a voxel is classified under any of the class ids, it will be considered part of that group. The groups will then show up in visualization, instead of the class names themselves.

An example is the following: you want to visualize all of the areas corresponding to "rubble" and "equipment". You could create a new group for rubble, and add various classes under it (such as "rocks", "bricks", "debris", etc), and do the same for equipment. Note that every class must be assigned to a group, and the groups must cover every class.

All of the voxseg computations are handled by modules/server.py. The voxseg methods themself are in voxel_world.py. The server is set up to listen on the following topics and services:

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

## Acknowledgements
Authors: Will Huey, Sean Brynjolfsson

We use OVSeg (https://github.com/facebookresearch/ov-seg) to obtain per-pixel image embeddings, and CLIP to obtain text embeddings (https://github.com/openai/CLIP). Both of these repos are contained in this one, with slight modifications to suit our needs.

For better rviz voxel visualization, we use rviz_lighting (from https://github.com/mogumbo/rviz_lighting.git).
