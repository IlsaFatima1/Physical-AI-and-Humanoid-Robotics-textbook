"""
Gazebo Simulation Integration Examples
Chapter 4: Gazebo Simulation Environment

This module demonstrates various Gazebo simulation concepts including:
- ROS 2 node integration with Gazebo
- Sensor data processing from simulated sensors
- Robot control in simulation environment
- TF transformations in simulated environment
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from tf2_geometry_msgs import do_transform_point
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import math
from cv_bridge import CvBridge
import cv2


class GazeboSimulationNode(Node):
    """
    Demonstrates integration between ROS 2 and Gazebo simulation.
    This node subscribes to simulated sensors and publishes commands
    to control the simulated robot.
    """

    def __init__(self):
        super().__init__('gazebo_simulation_node')

        # Initialize CV bridge for image processing
        self.cv_bridge = CvBridge()

        # Robot state variables
        self.current_pose = None
        self.current_twist = None
        self.laser_scan = None
        self.camera_image = None
        self.imu_data = None

        # TF broadcaster for publishing transforms
        self.tf_broadcaster = TransformBroadcaster(self)

        # Setup subscribers for simulated sensors
        self.setup_sensor_subscribers()

        # Setup publishers for robot control
        self.setup_control_publishers()

        # Setup timers for periodic tasks
        self.setup_timers()

        # Setup TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("Gazebo Simulation Node initialized")

    def setup_sensor_subscribers(self):
        """Setup subscribers for simulated sensors"""
        # Subscribe to laser scan data from simulated LiDAR
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/robot/scan',  # Standard Gazebo LiDAR topic
            self.scan_callback,
            10
        )

        # Subscribe to camera image data
        self.image_subscription = self.create_subscription(
            Image,
            '/robot/camera/image_raw',  # Standard Gazebo camera topic
            self.image_callback,
            10
        )

        # Subscribe to IMU data
        self.imu_subscription = self.create_subscription(
            Imu,
            '/robot/imu/data',  # Standard Gazebo IMU topic
            self.imu_callback,
            10
        )

        # Subscribe to odometry data
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/robot/odom',  # Standard Gazebo odometry topic
            self.odom_callback,
            10
        )

    def setup_control_publishers(self):
        """Setup publishers for robot control"""
        # Publisher for velocity commands
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/robot/cmd_vel',
            10
        )

        # Publisher for visualization markers
        self.marker_publisher = self.create_publisher(
            PointStamped,
            '/robot/safety_point',
            10
        )

    def setup_timers(self):
        """Setup timers for periodic tasks"""
        # Timer for processing sensor data and publishing commands
        self.control_timer = self.create_timer(
            0.1,  # 10 Hz
            self.control_loop
        )

        # Timer for TF publishing
        self.tf_timer = self.create_timer(
            0.05,  # 20 Hz
            self.publish_transforms
        )

        # Timer for sensor processing
        self.sensor_timer = self.create_timer(
            0.5,  # 2 Hz
            self.process_sensors
        )

    def scan_callback(self, msg):
        """Callback for laser scan data from simulated LiDAR"""
        self.laser_scan = msg
        self.get_logger().debug(f"Received laser scan with {len(msg.ranges)} points")

        # Process the scan data for obstacle detection
        self.detect_obstacles_in_scan(msg)

    def image_callback(self, msg):
        """Callback for camera image data from simulated camera"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.camera_image = cv_image

            # Process the image for object detection
            processed_image = self.process_camera_image(cv_image)

            # Log image dimensions
            self.get_logger().debug(f"Received camera image: {cv_image.shape}")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def imu_callback(self, msg):
        """Callback for IMU data from simulated IMU"""
        self.imu_data = msg
        self.get_logger().debug(f"Received IMU data - Linear acceleration: [{msg.linear_acceleration.x:.2f}, {msg.linear_acceleration.y:.2f}, {msg.linear_acceleration.z:.2f}]")

    def odom_callback(self, msg):
        """Callback for odometry data from simulated robot"""
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist
        self.get_logger().debug(f"Received odometry - Position: [{msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f}]")

    def detect_obstacles_in_scan(self, scan_msg):
        """Process laser scan to detect obstacles"""
        if len(scan_msg.ranges) == 0:
            return

        # Define minimum distance threshold for obstacle detection
        min_distance = 1.0  # meters
        obstacle_count = 0

        for i, range_val in enumerate(scan_msg.ranges):
            if not math.isnan(range_val) and range_val < min_distance:
                obstacle_count += 1

        if obstacle_count > 0:
            self.get_logger().info(f"Detected {obstacle_count} obstacles within {min_distance}m")

    def process_camera_image(self, image):
        """Process camera image from simulation"""
        # Example: Convert to grayscale and detect edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Example: Detect circles using Hough transform
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            min_dist=50,
            param1=50,
            param2=30,
            min_radius=10,
            max_radius=100
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)

        return image

    def control_loop(self):
        """Main control loop for the simulated robot"""
        if self.laser_scan is not None:
            # Simple obstacle avoidance based on laser scan
            cmd_vel = self.obstacle_avoidance_control()
            self.cmd_vel_publisher.publish(cmd_vel)

    def obstacle_avoidance_control(self):
        """Simple obstacle avoidance algorithm using laser scan data"""
        cmd_vel = Twist()

        if self.laser_scan is None:
            return cmd_vel

        # Get ranges from front, left, and right sectors
        ranges = self.laser_scan.ranges
        num_ranges = len(ranges)

        # Front sector (30 degrees in front)
        front_start = num_ranges // 2 - num_ranges // 24  # -15 degrees
        front_end = num_ranges // 2 + num_ranges // 24    # +15 degrees

        # Left sector
        left_start = num_ranges * 3 // 4 - num_ranges // 24
        left_end = num_ranges * 3 // 4 + num_ranges // 24

        # Right sector
        right_start = num_ranges // 4 - num_ranges // 24
        right_end = num_ranges // 4 + num_ranges // 24

        # Calculate minimum distances in each sector
        front_min = min(ranges[front_start:front_end]) if front_start < front_end else float('inf')
        left_min = min(ranges[left_start:left_end]) if left_start < left_end else float('inf')
        right_min = min(ranges[right_start:right_end]) if right_start < right_end else float('inf')

        # Simple control logic
        safe_distance = 1.0
        forward_speed = 0.5
        turn_speed = 0.5

        if front_min < safe_distance:
            # Obstacle in front, turn away
            if left_min > right_min:
                cmd_vel.angular.z = turn_speed  # Turn left
            else:
                cmd_vel.angular.z = -turn_speed  # Turn right
        else:
            # No obstacle in front, move forward
            cmd_vel.linear.x = forward_speed

        return cmd_vel

    def process_sensors(self):
        """Periodic sensor processing"""
        if self.imu_data is not None:
            # Example: Publish a safety point based on IMU data
            point_stamped = PointStamped()
            point_stamped.header = Header()
            point_stamped.header.stamp = self.get_clock().now().to_msg()
            point_stamped.header.frame_id = "robot_base_link"

            # Use IMU data to set a point (example: position based on acceleration)
            point_stamped.point.x = self.imu_data.linear_acceleration.x * 0.1
            point_stamped.point.y = self.imu_data.linear_acceleration.y * 0.1
            point_stamped.point.z = 0.0

            self.marker_publisher.publish(point_stamped)

    def publish_transforms(self):
        """Publish TF transforms for the simulated robot"""
        # Example: Publish a transform for a laser scanner relative to base link
        from geometry_msgs.msg import TransformStamped

        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "robot_base_link"
        t.child_frame_id = "robot_laser_link"

        # Set transform (example: laser is 0.1m in front of base, 0.05m up)
        t.transform.translation.x = 0.1
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.05

        # No rotation (identity quaternion)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

    def transform_point_example(self):
        """Example of transforming points between coordinate frames"""
        try:
            # Create a point in the laser frame
            point_in_laser = PointStamped()
            point_in_laser.header.frame_id = "robot_laser_link"
            point_in_laser.point.x = 1.0
            point_in_laser.point.y = 0.0
            point_in_laser.point.z = 0.0

            # Transform to base link frame
            transform = self.tf_buffer.lookup_transform(
                "robot_base_link",
                "robot_laser_link",
                rclpy.time.Time()
            )

            point_in_base = do_transform_point(point_in_laser, transform)
            self.get_logger().info(f"Point in base frame: [{point_in_base.point.x:.2f}, {point_in_base.point.y:.2f}, {point_in_base.point.z:.2f}]")

        except tf2_ros.TransformException as ex:
            self.get_logger().error(f"Transform failed: {ex}")

    def get_robot_state(self):
        """Get the current state of the robot from simulation"""
        state = {
            'position': None,
            'velocity': None,
            'laser_data': None,
            'imu_data': None
        }

        if self.current_pose:
            state['position'] = {
                'x': self.current_pose.position.x,
                'y': self.current_pose.position.y,
                'z': self.current_pose.position.z,
                'orientation': {
                    'x': self.current_pose.orientation.x,
                    'y': self.current_pose.orientation.y,
                    'z': self.current_pose.orientation.z,
                    'w': self.current_pose.orientation.w
                }
            }

        if self.current_twist:
            state['velocity'] = {
                'linear': {
                    'x': self.current_twist.linear.x,
                    'y': self.current_twist.linear.y,
                    'z': self.current_twist.linear.z
                },
                'angular': {
                    'x': self.current_twist.angular.x,
                    'y': self.current_twist.angular.y,
                    'z': self.current_twist.angular.z
                }
            }

        if self.laser_scan:
            state['laser_data'] = {
                'ranges': self.laser_scan.ranges,
                'intensities': self.laser_scan.intensities,
                'angle_min': self.laser_scan.angle_min,
                'angle_max': self.laser_scan.angle_max,
                'angle_increment': self.laser_scan.angle_increment
            }

        if self.imu_data:
            state['imu_data'] = {
                'linear_acceleration': {
                    'x': self.imu_data.linear_acceleration.x,
                    'y': self.imu_data.linear_acceleration.y,
                    'z': self.imu_data.linear_acceleration.z
                },
                'angular_velocity': {
                    'x': self.imu_data.angular_velocity.x,
                    'y': self.imu_data.angular_velocity.y,
                    'z': self.imu_data.angular_velocity.z
                }
            }

        return state


def main(args=None):
    """Main function to run the Gazebo simulation integration node"""
    rclpy.init(args=args)

    # Create the simulation node
    simulation_node = GazeboSimulationNode()

    try:
        # Spin the node to handle callbacks
        rclpy.spin(simulation_node)
    except KeyboardInterrupt:
        simulation_node.get_logger().info('Interrupted by user')
    finally:
        simulation_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()