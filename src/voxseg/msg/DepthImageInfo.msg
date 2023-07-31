# DepthImageInfo.msg
sensor_msgs/Image rgb_image
sensor_msgs/Image depth_image
float64[16] cam_extrinsics
float64[16] depth_extrinsics # only used if cams are unaligned