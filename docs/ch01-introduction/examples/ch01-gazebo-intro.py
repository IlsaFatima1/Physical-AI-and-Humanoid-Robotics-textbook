#!/usr/bin/env python3
"""
Chapter 1 Gazebo Example: Introduction to Simulation in Physical AI

This example demonstrates how Gazebo simulation is used in Physical AI and
humanoid robotics development. It shows the basic concepts of simulation
environments for testing and validating robotic algorithms before deployment
on real hardware.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Header
import time
import math


class GazeboSimulationNode(Node):
    """
    A ROS 2 node that demonstrates interaction with Gazebo simulation.
    This represents the simulation environment component mentioned in Chapter 1.
    """

    def __init__(self):
        super().__init__('ch01_gazebo_intro')

        # Publisher for joint states (simulating robot joint movements)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Publisher for robot velocity commands (for mobile base)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Timer for publishing messages
        self.timer = self.create_timer(0.1, self.publish_simulation_data)

        # Simulated joint names for a simple humanoid model
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint',
            'right_shoulder_joint', 'right_elbow_joint'
        ]

        # Initialize joint positions
        self.joint_positions = [0.0] * len(self.joint_names)
        self.time_step = 0.0

        self.get_logger().info('Gazebo simulation node initialized')

    def publish_simulation_data(self):
        """
        Publish simulated sensor and joint data to mimic Gazebo simulation output.
        This demonstrates how simulation provides sensor data for Physical AI systems.
        """
        # Create and publish joint states
        joint_msg = JointState()
        joint_msg.header = Header()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = self.joint_names

        # Simulate some joint movement (e.g., simple walking motion)
        self.time_step += 0.1
        for i in range(len(self.joint_positions)):
            # Create different movement patterns for different joint types
            if 'hip' in self.joint_names[i]:
                self.joint_positions[i] = 0.2 * math.sin(self.time_step)
            elif 'knee' in self.joint_names[i]:
                self.joint_positions[i] = 0.1 * math.sin(self.time_step + math.pi/2)
            elif 'ankle' in self.joint_names[i]:
                self.joint_positions[i] = 0.05 * math.sin(self.time_step + math.pi)
            elif 'shoulder' in self.joint_names[i]:
                self.joint_positions[i] = 0.15 * math.sin(self.time_step * 0.5)
            elif 'elbow' in self.joint_names[i]:
                self.joint_positions[i] = 0.1 * math.sin(self.time_step * 0.7)

        joint_msg.position = self.joint_positions
        joint_msg.velocity = [0.0] * len(self.joint_positions)
        joint_msg.effort = [0.0] * len(self.joint_positions)

        self.joint_pub.publish(joint_msg)

        # Publish some velocity commands (simulating mobile base movement)
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.1  # Move forward slowly
        cmd_msg.angular.z = 0.05 * math.sin(self.time_step * 0.3)  # Gentle turning
        self.cmd_vel_pub.publish(cmd_msg)

        # Log the current state
        self.get_logger().debug(f'Published joint states: {[f"{pos:.3f}" for pos in self.joint_positions[:3]]}...')


def main(args=None):
    """
    Main function demonstrating Gazebo simulation concepts from Chapter 1.
    """
    print("=== Chapter 1: Gazebo Simulation Introduction ===\n")

    print("Gazebo is a 3D simulation environment for robotics that provides:")
    print("• High-fidelity physics simulation")
    print("• Realistic rendering capabilities")
    print("• Sensor simulation for testing algorithms")
    print("• Safe environment for testing before real hardware deployment\n")

    rclpy.init(args=args)

    try:
        node = GazeboSimulationNode()

        print("Simulation node created. Publishing joint states and velocity commands...")
        print("This simulates how a humanoid robot would behave in a Gazebo environment.")
        print("In a real Gazebo setup, these commands would control a simulated robot model.\n")

        print("Key concepts demonstrated:")
        print("• Joint state publishing (mimicking real robot sensors)")
        print("• Velocity commands for mobile base control")
        print("• Physics simulation of robot movements")
        print("• Sensor data generation for perception algorithms\n")

        # Run for 10 seconds to demonstrate the simulation
        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < 10.0:
            rclpy.spin_once(node, timeout_sec=0.1)

        print("Simulation completed. In a real Gazebo environment:")
        print("• The robot model would be visualized in the Gazebo GUI")
        print("• Physics would simulate real-world forces and interactions")
        print("• Sensors would provide realistic data for testing algorithms")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()