# Orientation tracking using Unscented Kalman Filter
Using IMU data (accelerometer and gyroscope reading) an Unscented Kalman Filter was implemented to track the orientation in 3 dimensional space.The model parameters were deteremined using ground truth data provided by Vicon motion capture system.

### Results
The plots for the estimated orientation along with the ground truth data (Vicon) for each data set (1,2 and 3) are shown below.

Data set 1:

Data set 2:

Data set 3:


### Instructions
Change the value of data_num in estimate_rot.py to correspond to the data set number that you want to test. Run the estimate_rot function.

