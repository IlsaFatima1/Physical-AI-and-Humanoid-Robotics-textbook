#!/usr/bin/env python3
"""
Chapter 1 Examples: Physical AI and Humanoid Robotics Concepts

This file demonstrates fundamental concepts introduced in Chapter 1 of the textbook,
including basic robot state representation, sensor data processing, and simple
control concepts relevant to humanoid robotics.
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class RobotState:
    """
    Represents the state of a humanoid robot including position, orientation,
    and joint angles. This demonstrates the concept of proprioceptive sensing
    mentioned in Chapter 1.
    """
    # Position and orientation in 3D space
    position: Tuple[float, float, float]  # (x, y, z) in meters
    orientation: Tuple[float, float, float, float]  # Quaternion (w, x, y, z)

    # Joint angles for humanoid robot (simplified example)
    joint_angles: List[float]  # Angles in radians

    # Velocity information
    linear_velocity: Tuple[float, float, float]  # (vx, vy, vz) in m/s
    angular_velocity: Tuple[float, float, float]  # (wx, wy, wz) in rad/s


class SimpleHumanoidSimulator:
    """
    A simplified simulator demonstrating Physical AI concepts from Chapter 1.
    This simulator shows how a robot maintains awareness of its state and
    environment, which is fundamental to Physical AI.
    """

    def __init__(self):
        # Initialize robot in a standing position
        self.state = RobotState(
            position=(0.0, 0.0, 0.8),  # Standing at 0.8m height
            orientation=(1.0, 0.0, 0.0, 0.0),  # No rotation (quaternion)
            joint_angles=[0.0] * 20,  # 20 joints with zero angles initially
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0)
        )

        # Simulated sensor data
        self.imu_data = {
            'acceleration': (0.0, 0.0, 9.81),  # Gravity vector when standing
            'angular_velocity': (0.0, 0.0, 0.0),
            'orientation': (1.0, 0.0, 0.0, 0.0)
        }

        self.camera_data = {
            'objects_in_view': [],
            'depth_map': np.zeros((480, 640))  # Simplified depth map
        }

    def update_state(self, dt: float):
        """
        Update the robot's state based on control inputs and physics.
        This demonstrates the real-time operation characteristic of Physical AI.
        """
        # In a real implementation, this would integrate control commands
        # and apply physics simulation

        # For this example, just simulate minor adjustments to maintain balance
        self._balance_control(dt)

        # Update sensor data based on new state
        self._update_sensors()

    def _balance_control(self, dt: float):
        """
        Simple balance control to keep the robot upright.
        This demonstrates the stability challenge mentioned in Chapter 1.
        """
        # Check if the robot is tilting too much
        tilt_threshold = 0.1  # Radians

        # Get tilt from IMU data (simplified)
        current_tilt = abs(self.imu_data['orientation'][1])  # x-axis tilt

        if current_tilt > tilt_threshold:
            # Apply corrective joint movements to restore balance
            # This is a simplified representation of complex balance control
            self.state.joint_angles[0] += 0.01 * dt  # Ankle joint adjustment
            self.state.joint_angles[1] -= 0.01 * dt  # Opposite ankle adjustment

    def _update_sensors(self):
        """
        Update sensor data based on the current state.
        This demonstrates sensor fusion and perception systems.
        """
        # Update IMU data based on current state
        self.imu_data['acceleration'] = (
            self.state.linear_velocity[0] * 0.1,  # Simplified acceleration
            self.state.linear_velocity[1] * 0.1,
            9.81 + self.state.linear_velocity[2] * 0.1  # Gravity + vertical acceleration
        )

        # Update orientation based on angular velocity
        # This is a simplified representation
        self.imu_data['orientation'] = self.state.orientation

    def get_sensor_data(self) -> dict:
        """
        Return current sensor data, demonstrating the sensing system
        component of humanoid robots mentioned in Chapter 1.
        """
        return {
            'imu': self.imu_data,
            'camera': self.camera_data,
            'joint_positions': self.state.joint_angles
        }

    def execute_action(self, action: str, parameters: dict = None):
        """
        Execute a simple action, demonstrating the action component
        of Vision-Language-Action (VLA) models that will be covered later.
        """
        if action == "move_forward":
            self.state.position = (
                self.state.position[0] + 0.1,  # Move 0.1m forward
                self.state.position[1],
                self.state.position[2]  # Maintain height
            )
        elif action == "wave_arm":
            # Simplified arm waving motion
            self.state.joint_angles[5] = 0.5  # Shoulder joint
            self.state.joint_angles[6] = 0.3  # Elbow joint
        elif action == "turn_head":
            # Simplified head turning
            self.state.joint_angles[2] = parameters.get('angle', 0.2) if parameters else 0.2


def demonstrate_physical_ai_concepts():
    """
    Demonstrate key concepts from Chapter 1: Introduction to Physical AI.
    """
    print("=== Chapter 1: Introduction to Physical AI and Humanoid Robotics ===\n")

    # Create a simple humanoid simulator
    robot = SimpleHumanoidSimulator()

    print("1. Robot State Representation:")
    print(f"   Position: {robot.state.position}")
    print(f"   Orientation: {robot.state.orientation}")
    print(f"   Joint angles: {len(robot.state.joint_angles)} joints\n")

    print("2. Sensing System - Current Sensor Data:")
    sensor_data = robot.get_sensor_data()
    print(f"   IMU Acceleration: {sensor_data['imu']['acceleration']}")
    print(f"   Joint Positions: {sensor_data['joint_positions'][:5]}...")  # Show first 5
    print(f"   Objects in view: {len(sensor_data['camera']['objects_in_view'])}\n")

    print("3. Real-time Operation - Simulating robot state updates:")
    for i in range(5):
        robot.update_state(0.01)  # Update with 10ms time step
        print(f"   Update {i+1}: Position = {robot.state.position}, Tilt = {abs(robot.imu_data['orientation'][1]):.3f}")

    print("\n4. Action Execution - Demonstrating Physical AI in action:")
    robot.execute_action("wave_arm")
    print(f"   After waving arm - Joint angles: {robot.state.joint_angles[5:7]}")

    robot.execute_action("move_forward")
    print(f"   After moving forward - New position: {robot.state.position}")

    print("\n5. Safety Consideration - Balance Control:")
    print("   The simulator includes simple balance control to prevent falls,")
    print("   demonstrating the safety-critical nature of Physical AI systems.")

    # Show how sensor data is used for balance
    print(f"   Current tilt: {abs(robot.imu_data['orientation'][1]):.3f} radians")
    print(f"   Acceleration vector: {robot.imu_data['acceleration']}")


if __name__ == "__main__":
    demonstrate_physical_ai_concepts()