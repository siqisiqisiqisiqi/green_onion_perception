# green_onion_perception
## Table of contents

- Camera Calibration
- Run the project
- Others

## Camera Calibration

### Camera parameter transformation to the ROS frame
ROS frame is different from the opencv frame (the camera frame is different). 
To visualize the calibration result and convert the camera parameters in the ROS frame. 
Run "processed_point_cloud.py" code and set "OPTION" to "E1". 
"E1.npz" will be automatically created which defines the camera parameters in the ROS 
frame. 

### Rotate the Z axis in IRF to the opposite direction
Traditionally, z axis direction in IRF after calibration is downward. 
Run "processed_point_cloud.py" code and set "OPTION" to "E2".
"E2.npz" will be automatically created which converts the Z axis upward.