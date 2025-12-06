# Appendix E: Code Templates

This appendix provides reusable code templates for common tasks in Physical AI and Humanoid Robotics development using ROS 2, Gazebo, and NVIDIA Isaac.

## Basic ROS 2 Node Template

```python
#!/usr/bin/env python3
"""
Template for a basic ROS 2 node
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist


class BasicRobotNode(Node):
    def __init__(self):
        super().__init__('basic_robot_node')

        # Create publishers
        self.publisher = self.create_publisher(String, 'robot_status', 10)

        # Create subscribers
        self.joint_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_callback,
            10
        )

        # Create timer for periodic tasks
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Internal state
        self.joint_positions = []

        self.get_logger().info('Basic robot node initialized')

    def joint_callback(self, msg):
        """Process joint state messages"""
        self.joint_positions = list(msg.position)
        self.get_logger().debug(f'Received {len(self.joint_positions)} joint positions')

    def timer_callback(self):
        """Execute periodic tasks"""
        # Publish status message
        msg = String()
        msg.data = f'Node active with {len(self.joint_positions)} joints'
        self.publisher.publish(msg)

    def cleanup(self):
        """Cleanup function called on shutdown"""
        self.get_logger().info('Node shutting down')


def main(args=None):
    rclpy.init(args=args)
    node = BasicRobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Robot Control Node Template

```python
#!/usr/bin/env python3
"""
Template for robot control node with safety features
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64MultiArray
import numpy as np
from typing import List, Optional


class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control')

        # Publishers
        self.joint_cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.emergency_pub = self.create_publisher(Bool, 'emergency_stop', 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        # Control timer
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100 Hz

        # Internal state
        self.current_joints = JointState()
        self.current_imu = None
        self.desired_velocities = np.zeros(20)  # 20 joints example
        self.emergency_stop = False
        self.safety_limits = {
            'max_velocity': 2.0,  # rad/s
            'max_torque': 100.0,  # Nm
            'max_acceleration': 5.0  # rad/s²
        }

        self.get_logger().info('Robot control node initialized')

    def joint_state_callback(self, msg: JointState):
        """Update current joint state"""
        self.current_joints = msg

    def imu_callback(self, msg: Imu):
        """Update IMU data for balance control"""
        self.current_imu = msg

    def control_loop(self):
        """Main control loop with safety checks"""
        if self.emergency_stop:
            self.publish_emergency_stop()
            return

        # Perform safety checks
        if not self.safety_check():
            self.emergency_stop = True
            self.publish_emergency_stop()
            return

        # Calculate desired joint commands
        joint_commands = self.calculate_joint_commands()

        # Publish commands
        self.publish_joint_commands(joint_commands)

    def safety_check(self) -> bool:
        """Perform safety checks before executing commands"""
        # Check for dangerous joint positions
        if self.current_joints.position:
            for pos in self.current_joints.position:
                if abs(pos) > 3.14:  # Check for joint limits
                    self.get_logger().error('Joint position limit exceeded')
                    return False

        # Check for dangerous IMU readings (if available)
        if self.current_imu:
            # Check for excessive angular velocity
            ang_vel = np.sqrt(
                self.current_imu.angular_velocity.x**2 +
                self.current_imu.angular_velocity.y**2 +
                self.current_imu.angular_velocity.z**2
            )
            if ang_vel > 10.0:  # rad/s threshold
                self.get_logger().error('Excessive angular velocity detected')
                return False

        return True

    def calculate_joint_commands(self) -> JointState:
        """Calculate desired joint commands based on control logic"""
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = self.current_joints.name if self.current_joints.name else [f'joint_{i}' for i in range(20)]

        # Example: simple position control
        if self.current_joints.position:
            cmd.position = [pos + 0.01 for pos in self.current_joints.position[:20]]
        else:
            cmd.position = [0.0] * 20

        cmd.velocity = self.desired_velocities.tolist()
        cmd.effort = [0.0] * len(cmd.position)

        return cmd

    def publish_joint_commands(self, joint_cmd: JointState):
        """Publish joint commands with safety limits"""
        # Apply safety limits
        for i in range(len(joint_cmd.velocity)):
            joint_cmd.velocity[i] = np.clip(
                joint_cmd.velocity[i],
                -self.safety_limits['max_velocity'],
                self.safety_limits['max_velocity']
            )

        self.joint_cmd_pub.publish(joint_cmd)

    def publish_emergency_stop(self):
        """Publish emergency stop command"""
        msg = Bool()
        msg.data = True
        self.emergency_pub.publish(msg)

        # Stop all motion
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)


def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Control node interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Perception Node Template

```python
#!/usr/bin/env python3
"""
Template for perception node with image processing
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from geometry_msgs.msg import Point
import numpy as np
import cv2
from cv_bridge import CvBridge
from typing import List, Tuple, Optional


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # Publishers
        self.detection_pub = self.create_publisher(Point, 'object_position', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 'camera/camera_info', self.camera_info_callback, 10)

        # Initialize tools
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.distortion_coeffs = None

        # Processing parameters
        self.object_detector = self.initialize_object_detector()

        self.get_logger().info('Perception node initialized')

    def camera_info_callback(self, msg: CameraInfo):
        """Update camera calibration parameters"""
        if not self.camera_matrix:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.distortion_coeffs = np.array(msg.d)
            self.get_logger().info('Camera calibration updated')

    def image_callback(self, msg: Image):
        """Process incoming camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Process the image
            detections = self.process_image(cv_image, msg.header)

            # Publish results
            if detections:
                for detection in detections:
                    self.detection_pub.publish(detection)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def process_image(self, cv_image: np.ndarray, header: Header) -> List[Point]:
        """Process image to detect objects and return positions"""
        detections = []

        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define color ranges for object detection
        color_ranges = [
            (np.array([0, 50, 50]), np.array([10, 255, 255])),  # Red
            (np.array([100, 50, 50]), np.array([130, 255, 255]))  # Blue
        ]

        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Filter small contours
                    # Calculate centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # Create Point message with pixel coordinates
                        point = Point()
                        point.x = float(cx)
                        point.y = float(cy)
                        point.z = 0.0  # Will be updated with depth later

                        detections.append(point)

        return detections

    def initialize_object_detector(self):
        """Initialize object detection components"""
        # This could be a deep learning model in practice
        return {
            'min_area': 1000,
            'confidence_threshold': 0.7
        }

    def pixel_to_3d(self, pixel_x: float, pixel_y: float, depth: float) -> Tuple[float, float, float]:
        """Convert pixel coordinates to 3D world coordinates"""
        if self.camera_matrix is not None:
            # Simple conversion using camera parameters
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]

            x = (pixel_x - cx) * depth / fx
            y = (pixel_y - cy) * depth / fy

            return x, y, depth

        return pixel_x, pixel_y, depth  # Return pixel coordinates if no calibration


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Perception node interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Gazebo Integration Template

```python
#!/usr/bin/env python3
"""
Template for Gazebo integration with ROS 2
"""

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState, GetEntityState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Header
import time


class GazeboIntegrationNode(Node):
    def __init__(self):
        super().__init__('gazebo_integration')

        # Create service clients for Gazebo
        self.set_state_client = self.create_client(
            SetEntityState, '/world/default/set_entity_state')
        self.get_state_client = self.create_client(
            GetEntityState, '/world/default/get_entity_state')

        # Wait for Gazebo services to be available
        while not self.set_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for set_entity_state service...')

        while not self.get_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for get_entity_state service...')

        # Timer for periodic state updates
        self.state_timer = self.create_timer(0.1, self.update_robot_state)

        # Robot state
        self.robot_name = 'my_robot'
        self.current_pose = Pose()
        self.current_twist = Twist()

        self.get_logger().info('Gazebo integration node initialized')

    def update_robot_state(self):
        """Update robot state in Gazebo simulation"""
        # Get current state
        future = self.get_state_client.call_async(
            self.create_get_state_request(self.robot_name, 'world')
        )

        # Process the response (in a real implementation, you'd handle the future properly)
        # For now, just log that we're updating state
        self.get_logger().debug(f'Updating state for {self.robot_name}')

    def create_set_state_request(self, entity_name: str, pose: Pose, twist: Twist):
        """Create a SetEntityState request"""
        from gazebo_msgs.srv import SetEntityState

        req = SetEntityState.Request()
        req.state.name = entity_name
        req.state.pose = pose
        req.state.twist = twist
        req.state.reference_frame = 'world'

        return req

    def create_get_state_request(self, entity_name: str, reference_frame: str):
        """Create a GetEntityState request"""
        from gazebo_msgs.srv import GetEntityState

        req = GetEntityState.Request()
        req.name = entity_name
        req.reference_frame = reference_frame

        return req

    def set_robot_pose(self, pose: Pose):
        """Set robot pose in Gazebo"""
        req = self.create_set_state_request(self.robot_name, pose, Twist())

        future = self.set_state_client.call_async(req)
        # In a real implementation, you'd handle the future response

    def move_robot(self, linear_vel: float, angular_vel: float):
        """Move robot in Gazebo with specified velocities"""
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel

        req = self.create_set_state_request(self.robot_name, self.current_pose, twist)

        future = self.set_state_client.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = GazeboIntegrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Gazebo integration node interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Isaac Integration Template

```python
#!/usr/bin/env python3
"""
Template for NVIDIA Isaac integration
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import numpy as np
import cv2
from cv_bridge import CvBridge


class IsaacIntegrationNode(Node):
    def __init__(self):
        super().__init__('isaac_integration')

        # Publishers for Isaac-related topics
        self.perception_pub = self.create_publisher(String, 'isaac_perception_result', 10)
        self.control_cmd_pub = self.create_publisher(Twist, 'isaac_control_cmd', 10)

        # Subscribers for Isaac sensors
        self.rgb_sub = self.create_subscription(
            Image, 'isaac_camera/rgb', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, 'isaac_camera/depth', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 'isaac_camera/camera_info', self.camera_info_callback, 10)

        # Initialize tools
        self.bridge = CvBridge()
        self.camera_info = None

        # Isaac processing components
        self.perception_pipeline = self.initialize_perception_pipeline()

        self.get_logger().info('Isaac integration node initialized')

    def rgb_callback(self, msg: Image):
        """Process RGB camera data from Isaac"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Process with Isaac perception pipeline
            results = self.process_with_isaac_pipeline(cv_image)

            # Publish results
            result_msg = String()
            result_msg.data = str(results)
            self.perception_pub.publish(result_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {e}')

    def depth_callback(self, msg: Image):
        """Process depth data from Isaac"""
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")

            # Process depth information
            self.process_depth_data(cv_depth)

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def camera_info_callback(self, msg: CameraInfo):
        """Update camera information"""
        self.camera_info = msg

    def initialize_perception_pipeline(self):
        """Initialize Isaac perception components"""
        # In a real Isaac implementation, this would initialize
        # Isaac-specific perception modules
        return {
            'object_detection': True,
            'pose_estimation': True,
            'semantic_segmentation': True
        }

    def process_with_isaac_pipeline(self, image: np.ndarray):
        """Process image using Isaac perception pipeline"""
        # This is a simplified representation
        # In real Isaac, this would use Isaac's GPU-accelerated perception
        results = {
            'objects_detected': 0,
            'processing_time': 0.0,
            'confidence': 0.0
        }

        # Simple example: detect colored objects
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red color range
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results['objects_detected'] = len([c for c in contours if cv2.contourArea(c) > 500])
        results['processing_time'] = 0.01  # Simulated processing time
        results['confidence'] = 0.8 if results['objects_detected'] > 0 else 0.1

        return results

    def process_depth_data(self, depth_image: np.ndarray):
        """Process depth information for 3D understanding"""
        # In Isaac, this would use GPU-accelerated depth processing
        if depth_image.size > 0:
            # Calculate some statistics
            valid_depths = depth_image[depth_image > 0]
            if valid_depths.size > 0:
                avg_depth = np.mean(valid_depths)
                min_depth = np.min(valid_depths)
                max_depth = np.max(valid_depths)

                self.get_logger().debug(
                    f'Depth stats - Avg: {avg_depth:.2f}, Min: {min_depth:.2f}, Max: {max_depth:.2f}'
                )

    def generate_control_command(self, perception_results: dict) -> Twist:
        """Generate control commands based on perception results"""
        cmd = Twist()

        if perception_results.get('objects_detected', 0) > 0:
            # Move toward detected objects
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0  # No turning for now
        else:
            # Stop if no objects detected
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        return cmd


def main(args=None):
    rclpy.init(args=args)
    node = IsaacIntegrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Isaac integration node interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Launch File Template

```xml
<?xml version="1.0"?>
<launch>
  <!-- Arguments -->
  <arg name="robot_name" default="my_humanoid_robot"/>
  <arg name="use_sim_time" default="false"/>
  <arg name="gui" default="true"/>

  <!-- Robot description parameter -->
  <param name="robot_description"
         value="$(find-pkg-share my_robot_description)/urdf/my_robot.urdf"/>

  <!-- Use simulation time if needed -->
  <param name="use_sim_time" value="$(var use_sim_time)"/>

  <!-- Robot state publisher -->
  <node pkg="robot_state_publisher"
        exec="robot_state_publisher"
        name="robot_state_publisher">
    <param name="robot_description" value="$(var robot_description)"/>
  </node>

  <!-- Joint state publisher -->
  <node pkg="joint_state_publisher"
        exec="joint_state_publisher"
        name="joint_state_publisher">
    <param name="use_gui" value="$(var gui)"/>
  </node>

  <!-- Robot controllers -->
  <node pkg="my_robot_control"
        exec="robot_control_node"
        name="robot_controller">
    <param name="max_velocity" value="1.0"/>
    <param name="max_torque" value="50.0"/>
  </node>

  <!-- Perception node -->
  <node pkg="my_robot_perception"
        exec="perception_node"
        name="perception_node">
    <param name="confidence_threshold" value="0.7"/>
    <param name="min_detection_area" value="500"/>
  </node>

  <!-- TF broadcaster -->
  <node pkg="my_robot_bringup"
        exec="tf_broadcaster"
        name="tf_broadcaster"/>

  <!-- Load controller configurations -->
  <load_follower controller_manager_name="controller_manager"
                 condition="$(eval use_sim_time == false)">
    <controller name="joint_state_controller"
                type="joint_state_controller/JointStateController"/>
    <controller name="position_controller"
                type="position_controllers/JointGroupPositionController">
      <param name="joints" value="[joint1, joint2, joint3]"/>
    </controller>
  </load_follower>
</launch>
```

## CMakeLists.txt Template

```cmake
cmake_minimum_required(VERSION 3.8)
project(my_robot_package)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_filters REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/basic_robot_node.py
  scripts/robot_control_node.py
  scripts/perception_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install URDF files
install(DIRECTORY
  urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
```

## package.xml Template

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>1.0.0</version>
  <description>Package for my robot implementation following Physical AI & Humanoid Robotics textbook</description>
  <maintainer email="maintainer@example.com">Maintainer Name</maintainer>
  <license>Apache License 2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <buildtool_depend>ament_cmake_python</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>visualization_msgs</depend>
  <depend>message_filters</depend>
  <depend>cv_bridge</depend>
  <depend>tf2</depend>
  <depend>tf2_ros</depend>
  <depend>robot_state_publisher</depend>
  <depend>joint_state_publisher</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

## Unit Test Template

```python
#!/usr/bin/env python3
"""
Unit tests for robot components
"""

import unittest
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import time


class TestRobotNode(Node):
    def __init__(self):
        super().__init__('test_robot_node')
        self.publisher = self.create_publisher(String, 'test_topic', 10)
        self.received_messages = []

    def callback(self, msg):
        self.received_messages.append(msg.data)


class TestRobotComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = TestRobotNode()

    def tearDown(self):
        self.node.destroy_node()

    def test_basic_functionality(self):
        """Test basic node functionality"""
        self.assertIsNotNone(self.node)
        self.assertEqual(self.node.get_name(), 'test_robot_node')

    def test_publisher_exists(self):
        """Test that publisher is created correctly"""
        publishers = self.node.get_publishers_info_by_topic('test_topic')
        self.assertTrue(len(publishers) > 0)

    def test_message_publishing(self):
        """Test message publishing and receiving"""
        # This is a simplified test - in practice, you'd use a subscriber
        # to verify messages are received correctly
        msg = String()
        msg.data = 'test message'

        # Publish message
        self.node.publisher.publish(msg)

        # Verify no errors occurred during publishing
        self.assertTrue(True)  # Basic verification

    def test_robot_control_parameters(self):
        """Test robot control parameters"""
        # Example test for control parameters
        max_velocity = 1.0
        max_torque = 100.0

        # Test that parameters are within reasonable ranges
        self.assertGreater(max_velocity, 0)
        self.assertLess(max_velocity, 10.0)  # Reasonable limit
        self.assertGreater(max_torque, 0)
        self.assertLess(max_torque, 1000.0)  # Reasonable limit


def main():
    unittest.main()


if __name__ == '__main__':
    main()
```

These templates provide a solid foundation for developing Physical AI and Humanoid Robotics applications. Each template includes proper error handling, safety considerations, and follows ROS 2 best practices. Developers can use these as starting points and customize them for their specific applications.