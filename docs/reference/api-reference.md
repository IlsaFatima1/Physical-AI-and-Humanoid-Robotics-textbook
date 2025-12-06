# API Reference

This document provides detailed reference information for the ROS 2 interfaces, messages, and services used throughout the Physical AI & Humanoid Robotics textbook.

## Standard ROS 2 Message Types

### Sensor Messages

#### sensor_msgs/Image
**Description**: Raw image data from cameras or other image-forming sensors
- **Fields**:
  - `std_msgs/Header header`: Timestamp and coordinate frame ID
  - `uint32 height`: Image height in pixels
  - `uint32 width`: Image width in pixels
  - `string encoding`: Encoding format (e.g., "rgb8", "mono8", "bgr8")
  - `uint8 is_bigendian`: Endianness of image data
  - `uint32 step`: Row step in bytes
  - `uint8[] data`: Image data array

**Usage**: Used for camera feeds, depth images, and other visual sensors

#### sensor_msgs/JointState
**Description**: State of a set of joints at a given timestamp
- **Fields**:
  - `std_msgs/Header header`: Timestamp and coordinate frame ID
  - `string[] name`: Joint names
  - `float64[] position`: Joint positions (rad or m)
  - `float64[] velocity`: Joint velocities (rad/s or m/s)
  - `float64[] effort`: Joint efforts (Nm or N)

**Usage**: Used to communicate the current state of robot joints

#### sensor_msgs/Imu
**Description**: Inertial measurement unit data
- **Fields**:
  - `std_msgs/Header header`: Timestamp and coordinate frame ID
  - `geometry_msgs/Quaternion orientation`: Orientation as quaternion
  - `float64[9] orientation_covariance`: Covariance matrix for orientation
  - `geometry_msgs/Vector3 angular_velocity`: Angular velocity vector
  - `float64[9] angular_velocity_covariance`: Covariance matrix for angular velocity
  - `geometry_msgs/Vector3 linear_acceleration`: Linear acceleration vector
  - `float64[9] linear_acceleration_covariance`: Covariance matrix for linear acceleration

**Usage**: Used for balance control and orientation sensing in humanoid robots

### Geometry Messages

#### geometry_msgs/Twist
**Description**: Linear and angular velocity in free space
- **Fields**:
  - `geometry_msgs/Vector3 linear`: Linear velocity (x, y, z)
  - `geometry_msgs/Vector3 angular`: Angular velocity (x, y, z)

**Usage**: Used for sending velocity commands to robot bases

#### geometry_msgs/Pose
**Description**: Position and orientation in 3D space
- **Fields**:
  - `geometry_msgs/Point position`: Position vector (x, y, z)
  - `geometry_msgs/Quaternion orientation`: Orientation as quaternion (x, y, z, w)

**Usage**: Used for specifying target positions and orientations

## Custom Message Types for Humanoid Robotics

### humanoid_msgs/BalanceState
**Description**: Current balance state of a humanoid robot
- **Fields**:
  - `std_msgs/Header header`: Timestamp and coordinate frame ID
  - `float64 center_of_mass_x`: Center of mass X position (m)
  - `float64 center_of_mass_y`: Center of mass Y position (m)
  - `float64 center_of_mass_z`: Center of mass Z position (m)
  - `float64 stability_margin`: Stability margin (m)
  - `bool is_balanced`: True if robot is currently balanced
  - `string balance_status`: Current balance status ("stable", "unstable", "recovering")

**Usage**: Used to monitor and control humanoid balance

### humanoid_msgs/StepCommand
**Description**: Command for taking a single step
- **Fields**:
  - `std_msgs/Header header`: Timestamp and coordinate frame ID
  - `string foot`: Which foot to step with ("left", "right", "both")
  - `geometry_msgs/Point step_position`: Target position for the step
  - `float64 step_height`: Maximum height of the step (m)
  - `float64 step_duration`: Expected duration of the step (s)

**Usage**: Used for controlling humanoid locomotion

### humanoid_msgs/ManipulationGoal
**Description**: Goal for a manipulation task
- **Fields**:
  - `std_msgs/Header header`: Timestamp and coordinate frame ID
  - `string arm`: Which arm to use ("left", "right", "both")
  - `geometry_msgs/Pose target_pose`: Target pose for the end effector
  - `float64 grasp_force`: Force to apply when grasping (N)
  - `string object_name`: Name of the object to manipulate
  - `bool release_after_grasp`: Whether to release the object after grasping

**Usage**: Used for specifying manipulation tasks

## Standard ROS 2 Services

### std_srvs/Empty
**Description**: Simple service that takes no parameters and returns nothing
- **Request**: Empty
- **Response**: Empty
- **Usage**: Used for triggering actions like calibration, reset, or emergency stop

### std_srvs/SetBool
**Description**: Service to set a boolean parameter
- **Request**: `bool data` - Boolean value to set
- **Response**: `bool success` - Whether the operation was successful
- `string message` - Additional information about the result
- **Usage**: Used for enabling/disabling features or modes

### std_srvs/Trigger
**Description**: Service that triggers an action and reports success
- **Request**: Empty
- **Response**: `bool success` - Whether the operation was successful
- `string message` - Additional information about the result
- **Usage**: Used for triggering actions like calibration or data collection

## Custom Services for Humanoid Robotics

### humanoid_srvs/ComputeTrajectory
**Description**: Service to compute a trajectory for humanoid movement
- **Request**:
  - `geometry_msgs/Pose start_pose`: Starting pose
  - `geometry_msgs/Pose goal_pose`: Goal pose
  - `string trajectory_type`: Type of trajectory ("walking", "reaching", "balancing")
  - `float64 max_velocity`: Maximum allowed velocity (m/s)
  - `float64 max_acceleration`: Maximum allowed acceleration (m/s²)
- **Response**:
  - `bool success`: Whether trajectory computation was successful
  - `trajectory_msgs/JointTrajectory trajectory`: Computed joint trajectory
  - `string error_message`: Error message if computation failed
  - `float64 computation_time`: Time taken for computation (s)

**Usage**: Used to plan complex humanoid movements

### humanoid_srvs/ExecuteBehavior
**Description**: Service to execute a predefined behavior
- **Request**:
  - `string behavior_name`: Name of the behavior to execute
  - `std_msgs/Float64MultiArray parameters`: Parameters for the behavior
- **Response**:
  - `bool success`: Whether the behavior execution started successfully
  - `string execution_id`: ID for tracking the execution
  - `string error_message`: Error message if execution failed

**Usage**: Used to execute complex behaviors like "wave_hello", "sit_down", "stand_up"

## Action Interfaces

### control_msgs/FollowJointTrajectory
**Description**: Action to follow a joint trajectory with feedback
- **Goal**: `trajectory_msgs/JointTrajectory trajectory` - Desired joint trajectory
- **Result**: `control_msgs/FollowJointTrajectoryResult` - Execution result
- **Feedback**: `control_msgs/FollowJointTrajectoryFeedback` - Current progress

**Usage**: Used for precise joint control in manipulation and locomotion

### move_base_msgs/MoveBase
**Description**: Action to move the robot base to a goal position
- **Goal**: `geometry_msgs/PoseStamped target_pose` - Target pose for the robot
- **Result**: `move_base_msgs/MoveBaseResult` - Result of the navigation
- **Feedback**: `move_base_msgs/MoveBaseFeedback` - Current navigation status

**Usage**: Used for navigation in humanoid robots

## Standard Topics for Humanoid Robots

### Sensor Topics

#### /joint_states
- **Type**: `sensor_msgs/JointState`
- **Direction**: Publisher
- **Description**: Current state of all joints in the robot
- **Frequency**: 50-100 Hz for real-time control

#### /imu/data
- **Type**: `sensor_msgs/Imu`
- **Direction**: Publisher
- **Description**: IMU data for balance and orientation
- **Frequency**: 100-200 Hz for balance control

#### /camera/image_raw
- **Type**: `sensor_msgs/Image`
- **Direction**: Publisher
- **Description**: Raw camera image data
- **Frequency**: 15-30 Hz for perception tasks

### Command Topics

#### /cmd_vel
- **Type**: `geometry_msgs/Twist`
- **Direction**: Subscriber
- **Description**: Velocity commands for the robot base
- **Frequency**: 10-50 Hz for navigation

#### /joint_group_position_controller/commands
- **Type**: `std_msgs/Float64MultiArray`
- **Direction**: Subscriber
- **Description**: Position commands for joint groups
- **Frequency**: 50-100 Hz for control

## TF (Transforms) System

### Coordinate Frames

For humanoid robots, the following coordinate frames are commonly used:

- `base_link`: Robot's main body frame, typically at the center of the torso
- `base_footprint`: Frame on the ground directly below the robot's center of mass
- `left_foot`: Frame attached to the left foot
- `right_foot`: Frame attached to the right foot
- `left_hand`: Frame attached to the left hand/end effector
- `right_hand`: Frame attached to the right hand/end effector
- `head`: Frame attached to the robot's head
- `camera_link`: Frame attached to the camera
- `map`: World-fixed frame for navigation
- `odom`: Odometry-based frame for localization

### TF Tree Example

```
map
 └── odom
     └── base_footprint
         └── base_link
             ├── torso
             ├── head
             │   └── camera_link
             ├── left_leg
             │   └── left_foot
             ├── right_leg
             │   └── right_foot
             ├── left_arm
             │   └── left_hand
             └── right_arm
                 └── right_hand
```

## ROS 2 Parameters for Humanoid Robots

### Control Parameters

#### Balance Control
- `balance_controller.kp`: Proportional gain for balance control (default: 10.0)
- `balance_controller.kd`: Derivative gain for balance control (default: 1.0)
- `balance_controller.max_torque`: Maximum torque for balance control (default: 100.0)
- `balance_controller.com_height`: Expected center of mass height (default: 0.8)

#### Walking Control
- `walking_controller.step_height`: Maximum step height (default: 0.05)
- `walking_controller.step_duration`: Duration of each step (default: 1.0)
- `walking_controller.max_velocity`: Maximum walking velocity (default: 0.3)
- `walking_controller.foot_separation`: Distance between feet (default: 0.2)

#### Manipulation Control
- `manipulation_controller.max_force`: Maximum grasp force (default: 50.0)
- `manipulation_controller.max_velocity`: Maximum joint velocity (default: 1.0)
- `manipulation_controller.tolerance`: Position tolerance (default: 0.01)

### Perception Parameters

#### Camera Settings
- `camera.resolution_width`: Camera resolution width (default: 640)
- `camera.resolution_height`: Camera resolution height (default: 480)
- `camera.frame_rate`: Camera frame rate (default: 30)
- `camera.exposure_time`: Camera exposure time (default: 0.033)

#### Object Detection
- `object_detector.confidence_threshold`: Minimum detection confidence (default: 0.7)
- `object_detector.max_objects`: Maximum number of objects to detect (default: 10)
- `object_detector.min_distance`: Minimum detection distance (default: 0.2)
- `object_detector.max_distance`: Maximum detection distance (default: 3.0)

## Common ROS 2 Launch Files Structure

### Basic Humanoid Robot Launch
```xml
<launch>
  <!-- Robot state publisher -->
  <node pkg="robot_state_publisher" exec="robot_state_publisher" name="robot_state_publisher">
    <param name="robot_description" value="$(var robot_description)"/>
  </node>

  <!-- Joint state publisher -->
  <node pkg="joint_state_publisher" exec="joint_state_publisher" name="joint_state_publisher"/>

  <!-- IMU driver -->
  <node pkg="imu_driver" exec="imu_node" name="imu_node"/>

  <!-- Camera driver -->
  <node pkg="camera_driver" exec="camera_node" name="camera_node"/>

  <!-- Balance controller -->
  <node pkg="balance_controller" exec="balance_node" name="balance_node"/>
</launch>
```

## Common ROS 2 Commands for Humanoid Robotics

### Basic Commands
```bash
# List all topics
ros2 topic list

# Echo a topic
ros2 topic echo /joint_states

# Call a service
ros2 service call /reset_robot std_srvs/srv/Empty

# Run a launch file
ros2 launch my_robot bringup.launch.py
```

### Debugging Commands
```bash
# Monitor TF tree
ros2 run tf2_tools view_frames

# Check node connections
ros2 node info <node_name>

# Monitor parameter values
ros2 param list
ros2 param get <node_name> <param_name>
```

This API reference provides comprehensive information about the ROS 2 interfaces used in humanoid robotics applications. Developers should refer to this document when implementing humanoid robot systems to ensure proper message usage and interface compatibility.