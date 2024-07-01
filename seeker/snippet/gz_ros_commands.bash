#date: 2024-07-01T16:35:46Z
#url: https://api.github.com/gists/e51f1b63b44c2d3a4b250bff7599aa92
#owner: https://api.github.com/users/caguero

# External Gz Transport, external bridge, external ROS node
ros2 run rclcpp_components component_container
ros2 component load /ComponentManager ros_gz_bridge ros_gz_bridge::RosGzBridge -p config_file:=/home/caguero/ros_gz_ws/src/ros_gz/ros_gz_sim_demos/config/camera_bridge.yaml -p name:=bridge
ros2 run ros_gz_bridge ros_subscriber_node
ros2 run ros_gz_bridge publisher_node 

# (Gz Transport + bridge), external ROS node
ros2 run rclcpp_components component_container
ros2 component load /ComponentManager ros_gz_bridge ros_gz_bridge::RosGzBridge -p config_file:=/home/caguero/ros_gz_ws/src/ros_gz/ros_gz_sim_demos/config/camera_bridge.yaml -p name:=bridge
ros2 run ros_gz_bridge ros_subscriber_node
ros2 component load /ComponentManager ros_gz_bridge ros_gz_bridge::Publisher

# (Gz Transport + bridge + ROS node)
ros2 run rclcpp_components component_container
ros2 component load /ComponentManager ros_gz_bridge ros_gz_bridge::RosGzBridge -p config_file:=/home/caguero/ros_gz_ws/src/ros_gz/ros_gz_sim_demos/config/camera_bridge.yaml -p name:=bridge -e use_intra_process_comms:=true
ros2 component load /ComponentManager ros_gz_bridge ros_gz_bridge::RosSubscriber -e use_intra_process_comms:=true
ros2 component load /ComponentManager ros_gz_bridge ros_gz_bridge::Publisher -e use_intra_process_comms:=true