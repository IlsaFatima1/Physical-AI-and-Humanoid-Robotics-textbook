#!/usr/bin/env python3
"""
Chapter 1 Isaac Example: Introduction to NVIDIA Isaac in Physical AI

This example demonstrates the concepts of NVIDIA Isaac platform for robotics,
which is part of the Physical AI ecosystem mentioned in Chapter 1. While a full
Isaac simulation requires the Isaac platform, this example shows how Isaac
concepts integrate with ROS 2 for AI-powered robotics.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from std_msgs.msg import Header, Float32
import numpy as np
import cv2
from cv_bridge import CvBridge
import time


class IsaacSimulationNode(Node):
    """
    A ROS 2 node that demonstrates Isaac-like functionality for Physical AI.
    This represents the NVIDIA Isaac platform component mentioned in Chapter 1.
    """

    def __init__(self):
        super().__init__('ch01_isaac_intro')

        # Publishers for Isaac-like sensor data
        self.image_pub = self.create_publisher(Image, 'isaac_camera/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, 'isaac_camera/camera_info', 10)
        self.depth_pub = self.create_publisher(Image, 'isaac_depth/image_raw', 10)
        self.ai_processing_pub = self.create_publisher(Float32, 'isaac_ai/process_rate', 10)

        # Timer for publishing simulated Isaac data
        self.timer = self.create_timer(0.033, self.publish_isaac_data)  # ~30 Hz

        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()

        # Simulated camera parameters (typical for Isaac sensors)
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fov = 60  # degrees

        # Time tracking for simulation
        self.time_step = 0.0

        self.get_logger().info('Isaac simulation node initialized')

    def create_simulated_image(self):
        """
        Create a simulated camera image that mimics what Isaac sensors would provide.
        This demonstrates Isaac's perception capabilities mentioned in Chapter 1.
        """
        # Create a synthetic image with shapes that represent objects
        image = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)

        # Add some geometric shapes to represent objects in the environment
        center_x, center_y = self.camera_width // 2, self.camera_height // 2

        # Add a blue rectangle (could represent a table)
        cv2.rectangle(image, (center_x - 100, center_y - 50), (center_x + 100, center_y + 50), (255, 0, 0), -1)

        # Add a red circle (could represent an object to manipulate)
        cv2.circle(image, (center_x, center_y - 100), 40, (0, 0, 255), -1)

        # Add some random noise to make it more realistic
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return image

    def create_simulated_depth(self):
        """
        Create a simulated depth image that mimics Isaac depth sensors.
        This demonstrates Isaac's 3D perception capabilities.
        """
        depth = np.zeros((self.camera_height, self.camera_width), dtype=np.float32)

        # Create depth values based on the image content
        center_x, center_y = self.camera_width // 2, self.camera_height // 2

        # Set depth for different regions (in meters)
        # Rectangle area (table) - farther away
        depth[center_y - 50:center_y + 50, center_x - 100:center_x + 100] = 1.5

        # Circle area (object) - closer
        y, x = np.ogrid[:self.camera_height, :self.camera_width]
        mask = (x - center_x)**2 + (y - (center_y - 100))**2 <= 40**2
        depth[mask] = 0.8

        # Fill in background with default depth
        depth[depth == 0] = 2.0  # Background at 2 meters

        return depth

    def publish_isaac_data(self):
        """
        Publish simulated Isaac sensor data including camera and depth images.
        This demonstrates Isaac's sensor fusion and AI processing capabilities.
        """
        # Create simulated sensor data
        image = self.create_simulated_image()
        depth = self.create_simulated_depth()

        # Convert to ROS messages
        try:
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            image_msg.header = Header()
            image_msg.header.stamp = self.get_clock().now().to_msg()
            image_msg.header.frame_id = "isaac_camera_optical_frame"
            self.image_pub.publish(image_msg)

            depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
            depth_msg.header = Header()
            depth_msg.header.stamp = self.get_clock().now().to_msg()
            depth_msg.header.frame_id = "isaac_depth_optical_frame"
            self.depth_pub.publish(depth_msg)

            # Publish camera info
            camera_info_msg = CameraInfo()
            camera_info_msg.header = image_msg.header
            camera_info_msg.width = self.camera_width
            camera_info_msg.height = self.camera_height
            # Simple camera matrix (approximate for 60 degree FOV)
            fov_rad = np.deg2rad(self.camera_fov)
            focal_length = self.camera_width / (2 * np.tan(fov_rad / 2))
            camera_info_msg.k = [focal_length, 0.0, self.camera_width/2,
                                0.0, focal_length, self.camera_height/2,
                                0.0, 0.0, 1.0]
            self.camera_info_pub.publish(camera_info_msg)

            # Simulate AI processing rate (frames per second processed)
            process_rate = Float32()
            process_rate.data = 25.0 + 5.0 * np.sin(self.time_step)  # Variable rate
            self.ai_processing_pub.publish(process_rate)

        except Exception as e:
            self.get_logger().error(f'Error publishing Isaac data: {e}')

        # Update time step
        self.time_step += 0.033

        # Log some information
        if int(self.time_step * 30) % 30 == 0:  # Log every second
            self.get_logger().info(
                f'Published Isaac sensor data: '
                f'Image {self.camera_width}x{self.camera_height}, '
                f'AI processing rate: {process_rate.data:.1f} FPS'
            )


def demonstrate_isaac_concepts():
    """
    Demonstrate key concepts about NVIDIA Isaac from Chapter 1.
    """
    print("=== Chapter 1: NVIDIA Isaac Platform Introduction ===\n")

    print("NVIDIA Isaac is a robotics platform that provides:")
    print("• Hardware-accelerated AI for robotics applications")
    print("• Simulation tools (Isaac Sim) for testing and development")
    print("• ROS 2 integration for standard robotics frameworks")
    print("• GPU-accelerated perception and planning algorithms\n")

    print("Key Isaac Components:")
    print("• Isaac ROS: GPU-accelerated ROS 2 packages")
    print("• Isaac Sim: High-fidelity simulation environment")
    print("• Isaac Apps: Reference applications for common robotics tasks")
    print("• Isaac Lab: Framework for robot learning\n")

    print("In Physical AI and Humanoid Robotics, Isaac provides:")
    print("• Real-time perception using deep learning")
    print("• 3D sensing and environment understanding")
    print("• GPU acceleration for complex AI algorithms")
    print("• Integration with standard robotics frameworks like ROS 2\n")


def main(args=None):
    """
    Main function demonstrating Isaac concepts from Chapter 1.
    """
    demonstrate_isaac_concepts()

    rclpy.init(args=args)

    try:
        node = IsaacSimulationNode()

        print("Starting Isaac simulation node...")
        print("This simulates Isaac's sensor and AI processing capabilities.")
        print("In a real Isaac setup, these would connect to actual sensors and AI models.\n")

        print("What's being simulated:")
        print("• Camera image publishing (simulating Isaac perception)")
        print("• Depth image publishing (simulating 3D sensing)")
        print("• AI processing rate (simulating GPU-accelerated inference)\n")

        # Run for 15 seconds to demonstrate Isaac capabilities
        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < 15.0:
            rclpy.spin_once(node, timeout_sec=0.1)

        print("\nIsaac simulation completed. In a real Isaac environment:")
        print("• AI models would run on GPU for accelerated inference")
        print("• Perception algorithms would process sensor data in real-time")
        print("• The system would integrate with ROS 2 for robotics control")
        print("• Isaac Sim would provide high-fidelity physics simulation")

    except KeyboardInterrupt:
        print("\nIsaac simulation interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()