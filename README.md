# VoxSeg for Visual Navigation - ROS Repository

Will Huey, Sean Brynjolfsson

Contains OVSeg repo (from: https://github.com/facebookresearch/ov-seg)




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
- Define the world frame (for a default frame, run the following)
```bash
rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map world 5
```

- Start rviz:
```bash
rosrun rviz rviz
```
- Go to displays > global options > fixed frame, and change it from map to world
  - if world doesn't appear, the previous step was probably skipped
- run publish_markers whenever new markers come in. They should render with nice colors in rviz
- Limits: it starts getting pretty slow above 100x100x50 resolution (<1hz). It may be necessary to use a point cloud message for larger scenes


