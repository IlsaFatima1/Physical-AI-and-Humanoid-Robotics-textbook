#!/usr/bin/env python3
"""
Chapter 2 Examples: Fundamentals of Robotics and AI Integration

This file demonstrates the fundamental concepts of robotics and AI integration,
including the perception-action cycle, sensor processing, and basic control systems
as described in Chapter 2 of the textbook.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String, Header
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import threading
import math


@dataclass
class RobotState:
    """
    Represents the state of the robot including position, velocity, and sensor data.
    This demonstrates the concept of robot state management from Chapter 2.
    """
    position: Vector3
    orientation: float  # Heading in radians
    linear_velocity: float
    angular_velocity: float
    sensor_data: dict  # Dictionary to hold various sensor readings


class PerceptionModule:
    """
    A simplified perception module that processes sensor data.
    This demonstrates the perception component of the perception-action cycle.
    """
    def __init__(self):
        self.laser_ranges = []
        self.camera_image = None
        self.object_detected = False
        self.closest_obstacle_distance = float('inf')

    def process_laser_scan(self, laser_data: List[float]) -> dict:
        """
        Process laser scan data to detect obstacles and free space.
        This demonstrates basic sensor processing from Chapter 2.
        """
        if not laser_data:
            return {}

        # Find closest obstacle
        valid_ranges = [r for r in laser_data if 0.1 < r < 10.0]  # Filter invalid ranges
        if valid_ranges:
            self.closest_obstacle_distance = min(valid_ranges)
        else:
            self.closest_obstacle_distance = float('inf')

        # Detect obstacles in front (±30 degrees)
        front_indices = slice(len(laser_data)//2 - 30, len(laser_data)//2 + 30)
        front_ranges = laser_data[front_indices]
        front_obstacles = [r for r in front_ranges if 0.1 < r < 2.0]  # Obstacles within 2m
        self.object_detected = len(front_obstacles) > 0

        return {
            'closest_obstacle_distance': self.closest_obstacle_distance,
            'front_obstacle_count': len(front_obstacles),
            'object_detected': self.object_detected,
            'free_directions': self._find_free_directions(laser_data)
        }

    def _find_free_directions(self, laser_data: List[float]) -> List[Tuple[int, float]]:
        """
        Find directions with no close obstacles.
        This demonstrates simple spatial reasoning from AI in robotics.
        """
        free_dirs = []
        for i, distance in enumerate(laser_data):
            if distance > 2.0:  # Consider direction free if obstacle > 2m away
                angle = i * (2 * math.pi / len(laser_data)) - math.pi  # Convert to angle
                free_dirs.append((i, angle))
        return free_dirs


class ReasoningModule:
    """
    A simplified reasoning module that makes decisions based on perception data.
    This demonstrates the reasoning component of the perception-action cycle.
    """
    def __init__(self):
        self.safety_distance = 0.5  # Minimum safe distance to obstacles
        self.goal_reached = False

    def make_decision(self, perception_data: dict, current_state: RobotState) -> dict:
        """
        Make navigation decisions based on perception data and current state.
        This demonstrates basic decision-making in AI-robotics integration.
        """
        decision = {
            'action': 'stop',  # Default action
            'linear_velocity': 0.0,
            'angular_velocity': 0.0,
            'confidence': 0.0
        }

        # Check if we have valid perception data
        if 'closest_obstacle_distance' not in perception_data:
            return decision

        closest_dist = perception_data['closest_obstacle_distance']

        # Simple navigation logic
        if closest_dist < self.safety_distance:
            # Too close to obstacle - turn away
            decision['action'] = 'turn_away'
            decision['linear_velocity'] = 0.0
            decision['angular_velocity'] = 0.5  # Turn right
            decision['confidence'] = 0.9
        elif perception_data.get('object_detected', False):
            # Object detected ahead - turn to avoid
            decision['action'] = 'avoid_object'
            decision['linear_velocity'] = 0.0
            decision['angular_velocity'] = 0.3
            decision['confidence'] = 0.8
        else:
            # Path is clear - move forward
            decision['action'] = 'move_forward'
            decision['linear_velocity'] = 0.3  # Move at 0.3 m/s
            decision['angular_velocity'] = 0.0
            decision['confidence'] = 0.7

        return decision


class ActionModule:
    """
    A simplified action module that executes commands.
    This demonstrates the action component of the perception-action cycle.
    """
    def __init__(self):
        self.current_command = Twist()
        self.is_executing = False

    def execute_command(self, decision: dict) -> Twist:
        """
        Execute the decision by creating appropriate velocity commands.
        This demonstrates the action component of AI-robotics integration.
        """
        command = Twist()
        command.linear.x = decision.get('linear_velocity', 0.0)
        command.angular.z = decision.get('angular_velocity', 0.0)

        self.current_command = command
        self.is_executing = True

        return command

    def stop_robot(self) -> Twist:
        """Stop the robot by sending zero velocities."""
        command = Twist()
        command.linear.x = 0.0
        command.angular.z = 0.0
        self.current_command = command
        self.is_executing = False
        return command


class RoboticsAIIntegrationNode(Node):
    """
    A ROS 2 node that demonstrates the integration of robotics and AI concepts.
    This implements the perception-action cycle described in Chapter 2.
    """
    def __init__(self):
        super().__init__('robotics_ai_integration')

        # Initialize the three modules of the perception-action cycle
        self.perception = PerceptionModule()
        self.reasoning = ReasoningModule()
        self.action = ActionModule()

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)
        self.status_pub = self.create_publisher(String, 'robot_status', 10)

        # Robot state
        self.robot_state = RobotState(
            position=Vector3(x=0.0, y=0.0, z=0.0),
            orientation=0.0,
            linear_velocity=0.0,
            angular_velocity=0.0,
            sensor_data={}
        )

        # Control timer for the perception-action cycle
        self.control_timer = self.create_timer(0.1, self.perception_action_cycle)

        # Internal state
        self.latest_scan = None
        self.cycle_count = 0

        self.get_logger().info('Robotics AI Integration node initialized')

    def scan_callback(self, msg: LaserScan):
        """Process incoming laser scan data."""
        self.latest_scan = msg.ranges

    def perception_action_cycle(self):
        """
        Execute one cycle of the perception-action loop.
        This demonstrates the fundamental cycle in AI-robotics integration.
        """
        # PERCEPTION: Process sensor data
        perception_data = {}
        if self.latest_scan:
            perception_data = self.perception.process_laser_scan(list(self.latest_scan))

        # REASONING: Make decisions based on perception
        decision = self.reasoning.make_decision(perception_data, self.robot_state)

        # ACTION: Execute the decision
        command = self.action.execute_command(decision)

        # Publish the command
        self.cmd_vel_pub.publish(command)

        # Update robot state (simplified)
        self.robot_state.linear_velocity = command.linear.x
        self.robot_state.angular_velocity = command.angular.z

        # Publish status
        status_msg = String()
        status_msg.data = f"Cycle {self.cycle_count}: {decision['action']} (confidence: {decision['confidence']:.2f})"
        self.status_pub.publish(status_msg)

        self.cycle_count += 1

        # Log the current cycle
        if self.cycle_count % 10 == 0:  # Log every 10 cycles
            self.get_logger().info(
                f"Perception-Action Cycle {self.cycle_count}: "
                f"Action={decision['action']}, "
                f"Linear Vel={command.linear.x:.2f}, "
                f"Angular Vel={command.angular.z:.2f}, "
                f"Closest Obstacle={self.perception.closest_obstacle_distance:.2f}m"
            )


def demonstrate_robotics_ai_concepts():
    """
    Demonstrate key concepts from Chapter 2: Fundamentals of Robotics and AI Integration.
    """
    print("=== Chapter 2: Fundamentals of Robotics and AI Integration ===\n")

    print("1. Perception-Action Cycle:")
    print("   - Perception: Sensing and interpreting the environment")
    print("   - Reasoning: Making decisions based on sensor data and goals")
    print("   - Action: Executing physical actions based on decisions")
    print("   - Learning: Updating models based on outcomes (simulated)\n")

    print("2. Core Components:")
    print("   - Mechanical Structure: Physical framework of the robot")
    print("   - Actuation Systems: Motors and mechanisms for movement")
    print("   - Sensing Systems: Sensors for environment perception")
    print("   - Control Systems: Processing and decision-making components\n")

    print("3. AI Techniques in Robotics:")
    print("   - Computer Vision: Object recognition and scene understanding")
    print("   - Machine Learning: Improving performance through experience")
    print("   - Natural Language Processing: Human-robot interaction")
    print("   - Path Planning: Developing action sequences to achieve goals\n")

    print("4. Integration Challenges:")
    print("   - Real-time Requirements: AI algorithms must operate within robot time constraints")
    print("   - Uncertainty Management: Dealing with noisy sensors and uncertain environments")
    print("   - Embodiment: AI must account for physical constraints and dynamics")
    print("   - Safety: Ensuring safe operation in human environments\n")

    print("This example demonstrates a simple robot that:")
    print("   - Perceives obstacles using simulated laser scan data")
    print("   - Reasons about the safest navigation strategy")
    print("   - Acts by sending velocity commands to move the robot")
    print("   - Operates in a continuous perception-action cycle")


class SimpleRobotSimulator:
    """
    A simple robot simulator to demonstrate the concepts without hardware.
    """
    def __init__(self):
        self.position = [0.0, 0.0]  # x, y position
        self.orientation = 0.0  # heading in radians
        self.velocity = [0.0, 0.0]  # linear velocity components
        self.angular_velocity = 0.0  # rotational velocity

        # Simulate some obstacles in the environment
        self.obstacles = [
            (2.0, 1.0, 0.3),  # (x, y, radius) of obstacle
            (3.0, -1.0, 0.4),
            (-1.0, 2.0, 0.2)
        ]

    def simulate_laser_scan(self) -> List[float]:
        """
        Simulate laser scan data based on robot position and obstacles.
        """
        scan_ranges = []
        num_rays = 360  # 1 degree resolution

        for i in range(num_rays):
            angle = (i * 2 * math.pi / num_rays) + self.orientation
            ray_x = math.cos(angle)
            ray_y = math.sin(angle)

            # Find distance to nearest obstacle in this direction
            min_distance = 10.0  # Max range
            for obs_x, obs_y, obs_radius in self.obstacles:
                # Calculate distance from robot to obstacle in ray direction
                dx = obs_x - self.position[0]
                dy = obs_y - self.position[1]

                # Project obstacle onto ray direction
                projection = dx * ray_x + dy * ray_y

                if projection > 0:  # Only consider obstacles in front
                    perpendicular_distance = math.sqrt(dx*dx + dy*dy - projection*projection)

                    if perpendicular_distance < obs_radius:
                        # Ray intersects obstacle
                        distance_to_surface = projection - math.sqrt(obs_radius*obs_radius - perpendicular_distance*perpendicular_distance)
                        if distance_to_surface > 0 and distance_to_surface < min_distance:
                            min_distance = distance_to_surface

            scan_ranges.append(min_distance)

        return scan_ranges

    def update_position(self, linear_vel: float, angular_vel: float, dt: float = 0.1):
        """
        Update robot position based on velocities.
        """
        # Update orientation
        self.orientation += angular_vel * dt

        # Update position based on linear velocity
        self.velocity[0] = linear_vel * math.cos(self.orientation)
        self.velocity[1] = linear_vel * math.sin(self.orientation)

        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt

    def get_state(self) -> RobotState:
        """
        Get the current robot state.
        """
        state = RobotState(
            position=Vector3(x=self.position[0], y=self.position[1], z=0.0),
            orientation=self.orientation,
            linear_velocity=math.sqrt(self.velocity[0]**2 + self.velocity[1]**2),
            angular_velocity=self.angular_velocity,
            sensor_data={}
        )
        return state


def run_simulated_demo():
    """
    Run a simulated demonstration of the perception-action cycle.
    """
    print("\n=== Simulated Perception-Action Cycle Demo ===\n")

    # Initialize components
    perception = PerceptionModule()
    reasoning = ReasoningModule()
    action = ActionModule()
    robot_sim = SimpleRobotSimulator()

    print("Starting simulated robot operation for 20 cycles...")
    print("The robot will navigate while avoiding simulated obstacles.\n")

    for cycle in range(20):
        # PERCEPTION: Get simulated sensor data
        simulated_scan = robot_sim.simulate_laser_scan()
        perception_data = perception.process_laser_scan(simulated_scan)

        # REASONING: Make navigation decision
        current_state = robot_sim.get_state()
        decision = reasoning.make_decision(perception_data, current_state)

        # ACTION: Execute command and update simulation
        command = action.execute_command(decision)
        robot_sim.update_position(command.linear.x, command.angular.z, 0.1)

        # Display information
        print(f"Cycle {cycle+1:2d}: Action={decision['action']:12s} | "
              f"Vel=({command.linear.x:4.2f}, {command.angular.z:5.2f}) | "
              f"Closest Obstacle={perception.closest_obstacle_distance:5.2f}m | "
              f"Pos=({robot_sim.position[0]:5.2f}, {robot_sim.position[1]:5.2f})")

        # Add a small delay to simulate real-time operation
        time.sleep(0.1)

    print(f"\nDemo completed. Final position: ({robot_sim.position[0]:.2f}, {robot_sim.position[1]:.2f})")


def main(args=None):
    """
    Main function demonstrating robotics and AI integration concepts.
    """
    demonstrate_robotics_ai_concepts()

    print("\n" + "="*60)
    run_simulated_demo()

    print("\n" + "="*60)
    print("Starting ROS 2 node for robotics AI integration...")
    print("This node implements the perception-action cycle with real ROS 2 interfaces.")
    print("In a real system, this would connect to actual robot hardware and sensors.\n")

    rclpy.init(args=args)

    try:
        node = RoboticsAIIntegrationNode()
        print("Node created. The perception-action cycle is now running.")
        print("The robot will continuously perceive its environment, reason about")
        print("navigation decisions, and act accordingly.\n")

        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\nRobotics AI integration node interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()