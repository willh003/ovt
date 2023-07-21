gnome-terminal --tab -- bash -c 'roscore & exec bash'
gnome-terminal --tab -- bash -c 'catkin_make && source devel/setup.bash && roslaunch voxseg rviz.launch & exec bash'
gnome-terminal --tab -- bash -c 'source devel/setup.bash && roslaunch voxseg server.launch & exec bash'
gnome-terminal --tab -- bash -c 'source devel/setup.bash && roslaunch rviz_lighting rviz_lighting.launch & exec bash'

source devel/setup.bash
