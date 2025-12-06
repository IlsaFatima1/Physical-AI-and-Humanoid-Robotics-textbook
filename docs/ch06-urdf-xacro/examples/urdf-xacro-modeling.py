"""
URDF and XACRO Modeling Examples
Chapter 6: URDF and XACRO for Robot Modeling

This module demonstrates various URDF and XACRO concepts including:
- Robot model validation and analysis
- Dynamic URDF generation
- Robot state publishing
- Model inspection utilities
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import xml.etree.ElementTree as ET
import math
import numpy as np
from std_msgs.msg import String


class URDFAnalyzer(Node):
    """
    Analyzes URDF models and provides information about robot structure,
    kinematics, and properties.
    """

    def __init__(self):
        super().__init__('urdf_analyzer')

        # Robot model data
        self.robot_tree = None
        self.links = {}
        self.joints = {}
        self.kinematic_chain = {}

        # Setup publishers
        self.model_info_pub = self.create_publisher(
            String,
            '/robot_model_info',
            10
        )

        # Setup timer for analysis
        self.analysis_timer = self.create_timer(5.0, self.analyze_model)

        # Load and analyze a sample URDF
        self.load_sample_urdf()

        self.get_logger().info("URDF Analyzer initialized")

    def load_sample_urdf(self):
        """Load and parse a sample URDF model"""
        # This would typically load from a file or parameter server
        # For demonstration, we'll create a simple model programmatically
        self.robot_tree = self.create_sample_robot_model()
        self.parse_urdf_model(self.robot_tree)

    def create_sample_robot_model(self):
        """Create a sample robot model programmatically"""
        # This simulates loading and parsing a URDF file
        robot = {
            'name': 'sample_robot',
            'links': {
                'base_link': {
                    'mass': 10.0,
                    'inertia': {'ixx': 0.416, 'iyy': 0.583, 'izz': 0.194, 'ixy': 0, 'ixz': 0, 'iyz': 0},
                    'visual': {'type': 'box', 'size': [0.5, 0.4, 0.2]},
                    'collision': {'type': 'box', 'size': [0.5, 0.4, 0.2]}
                },
                'left_wheel': {
                    'mass': 0.5,
                    'inertia': {'ixx': 0.001, 'iyy': 0.001, 'izz': 0.002, 'ixy': 0, 'ixz': 0, 'iyz': 0},
                    'visual': {'type': 'cylinder', 'radius': 0.05, 'length': 0.02},
                    'collision': {'type': 'cylinder', 'radius': 0.05, 'length': 0.02}
                },
                'right_wheel': {
                    'mass': 0.5,
                    'inertia': {'ixx': 0.001, 'iyy': 0.001, 'izz': 0.002, 'ixy': 0, 'ixz': 0, 'iyz': 0},
                    'visual': {'type': 'cylinder', 'radius': 0.05, 'length': 0.02},
                    'collision': {'type': 'cylinder', 'radius': 0.05, 'length': 0.02}
                }
            },
            'joints': {
                'left_wheel_joint': {
                    'type': 'continuous',
                    'parent': 'base_link',
                    'child': 'left_wheel',
                    'origin': {'xyz': [0.15, 0.25, -0.1], 'rpy': [0, 0, 0]},
                    'axis': [0, 1, 0]
                },
                'right_wheel_joint': {
                    'type': 'continuous',
                    'parent': 'base_link',
                    'child': 'right_wheel',
                    'origin': {'xyz': [0.15, -0.25, -0.1], 'rpy': [0, 0, 0]},
                    'axis': [0, 1, 0]
                }
            }
        }

        return robot

    def parse_urdf_model(self, robot_model):
        """Parse URDF model data into internal structures"""
        self.links = robot_model['links']
        self.joints = robot_model['joints']

        # Build kinematic chain
        self.kinematic_chain = {}
        for joint_name, joint_data in self.joints.items():
            parent = joint_data['parent']
            child = joint_data['child']

            if parent not in self.kinematic_chain:
                self.kinematic_chain[parent] = []
            self.kinematic_chain[parent].append(child)

        # Find base link (link with no parent)
        all_children = set()
        for joint_data in self.joints.values():
            all_children.add(joint_data['child'])

        base_links = set(self.links.keys()) - all_children
        self.base_link = next(iter(base_links)) if base_links else 'base_link'

    def analyze_model(self):
        """Analyze the URDF model and publish information"""
        if not self.links or not self.joints:
            return

        analysis_info = self.generate_model_analysis()
        analysis_msg = String()
        analysis_msg.data = analysis_info
        self.model_info_pub.publish(analysis_msg)

        self.get_logger().info(f"Model analysis:\n{analysis_info}")

    def generate_model_analysis(self):
        """Generate detailed analysis of the robot model"""
        info = []
        info.append(f"Robot Model Analysis for: {self.robot_tree['name']}")
        info.append(f"Base Link: {self.base_link}")
        info.append(f"Number of Links: {len(self.links)}")
        info.append(f"Number of Joints: {len(self.joints)}")
        info.append("")

        # Link analysis
        info.append("Link Analysis:")
        for link_name, link_data in self.links.items():
            info.append(f"  Link: {link_name}")
            info.append(f"    Mass: {link_data['mass']} kg")
            info.append(f"    Inertia: [{link_data['inertia']['ixx']:.3f}, {link_data['inertia']['iyy']:.3f}, {link_data['inertia']['izz']:.3f}]")
            info.append(f"    Visual: {link_data['visual']['type']}")
            info.append(f"    Collision: {link_data['collision']['type']}")
            info.append("")

        # Joint analysis
        info.append("Joint Analysis:")
        for joint_name, joint_data in self.joints.items():
            info.append(f"  Joint: {joint_name}")
            info.append(f"    Type: {joint_data['type']}")
            info.append(f"    Parent: {joint_data['parent']}")
            info.append(f"    Child: {joint_data['child']}")
            info.append(f"    Origin: {joint_data['origin']['xyz']}")
            info.append(f"    Axis: {joint_data['axis']}")
            info.append("")

        # Kinematic chain analysis
        info.append("Kinematic Chain:")
        self.analyze_kinematic_chain(self.base_link, info, 0)

        return "\n".join(info)

    def analyze_kinematic_chain(self, link, info, depth):
        """Recursively analyze kinematic chain"""
        indent = "  " * depth
        info.append(f"{indent}{link}")

        if link in self.kinematic_chain:
            for child in self.kinematic_chain[link]:
                self.analyze_kinematic_chain(child, info, depth + 1)


class XacroProcessor(Node):
    """
    Simulates XACRO processing capabilities for parameterized robot models.
    """

    def __init__(self):
        super().__init__('xacro_processor')

        # Publisher for processed URDF
        self.urdf_pub = self.create_publisher(
            String,
            '/processed_urdf',
            10
        )

        # Setup timer for processing demonstration
        self.process_timer = self.create_timer(10.0, self.process_xacro_demo)

        self.get_logger().info("XACRO Processor initialized")

    def process_xacro_demo(self):
        """Demonstrate XACRO processing with parameter substitution"""
        # Define parameters similar to XACRO properties
        params = {
            'robot_name': 'my_robot',
            'wheel_radius': 0.05,
            'wheel_width': 0.02,
            'base_length': 0.5,
            'base_width': 0.4,
            'base_height': 0.2,
            'pi': math.pi
        }

        # Generate URDF with parameter substitution
        urdf_content = self.generate_parameterized_urdf(params)

        # Publish the processed URDF
        urdf_msg = String()
        urdf_msg.data = urdf_content
        self.urdf_pub.publish(urdf_msg)

        self.get_logger().info("Published parameterized URDF with XACRO-like processing")

    def generate_parameterized_urdf(self, params):
        """Generate URDF with parameter substitution (simulating XACRO)"""
        # This simulates what XACRO does - substitute parameters in URDF template
        urdf_template = f"""<?xml version="1.0"?>
<robot name="{params['robot_name']}">
  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.416" ixy="0" ixz="0" iyy="0.583" iyz="0" izz="0.194"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{params['base_length']} {params['base_width']} {params['base_height']}"/>
      </geometry>
      <material name="orange">
        <color rgba="1 0.5 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{params['base_length']} {params['base_width']} {params['base_height']}"/>
      </geometry>
    </collision>
  </link>

  <!-- Left wheel joint -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="{params['base_length']/2 - params['wheel_width']/2} {params['base_width']/2 + params['wheel_radius']} -{params['base_height']/2}" rpy="{-params['pi']/2} 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <!-- Left wheel link -->
  <link name="left_wheel">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="{params['wheel_radius']}" length="{params['wheel_width']}"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="{params['wheel_radius']}" length="{params['wheel_width']}"/>
      </geometry>
    </collision>
  </link>

  <!-- Right wheel joint -->
  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="{params['base_length']/2 - params['wheel_width']/2} {-params['base_width']/2 - params['wheel_radius']} -{params['base_height']/2}" rpy="{-params['pi']/2} 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <!-- Right wheel link -->
  <link name="right_wheel">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="{params['wheel_radius']}" length="{params['wheel_width']}"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="{params['wheel_radius']}" length="{params['wheel_width']}"/>
      </geometry>
    </collision>
  </link>
</robot>"""

        return urdf_template


class RobotStatePublisher(Node):
    """
    Publishes TF transforms based on joint states, simulating robot_state_publisher.
    """

    def __init__(self):
        super().__init__('robot_state_publisher')

        # Initialize joint positions
        self.joint_positions = {
            'left_wheel_joint': 0.0,
            'right_wheel_joint': 0.0
        }

        # Setup subscribers and publishers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Setup TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Setup timer for publishing transforms
        self.publish_timer = self.create_timer(0.05, self.publish_transforms)  # 20 Hz

        self.get_logger().info("Robot State Publisher initialized")

    def joint_state_callback(self, msg):
        """Update joint positions from joint state messages"""
        for i, name in enumerate(msg.name):
            if name in self.joint_positions:
                self.joint_positions[name] = msg.position[i]

    def publish_transforms(self):
        """Publish TF transforms for the robot"""
        transforms = []

        # Base to left wheel transform
        left_wheel_transform = TransformStamped()
        left_wheel_transform.header.stamp = self.get_clock().now().to_msg()
        left_wheel_transform.header.frame_id = 'base_link'
        left_wheel_transform.child_frame_id = 'left_wheel'

        # Position: wheel offset from base
        left_wheel_transform.transform.translation.x = 0.15
        left_wheel_transform.transform.translation.y = 0.25
        left_wheel_transform.transform.translation.z = -0.1

        # Rotation: account for wheel rotation around Y axis
        wheel_angle = self.joint_positions['left_wheel_joint']
        cy = math.cos(wheel_angle / 2.0)
        sy = math.sin(wheel_angle / 2.0)
        cp = math.cos(-math.pi / 4)  # Account for initial wheel orientation
        sp = math.sin(-math.pi / 4)

        left_wheel_transform.transform.rotation.w = cy * cp
        left_wheel_transform.transform.rotation.x = cy * sp
        left_wheel_transform.transform.rotation.y = sy * cp
        left_wheel_transform.transform.rotation.z = sy * sp

        transforms.append(left_wheel_transform)

        # Base to right wheel transform
        right_wheel_transform = TransformStamped()
        right_wheel_transform.header.stamp = self.get_clock().now().to_msg()
        right_wheel_transform.header.frame_id = 'base_link'
        right_wheel_transform.child_frame_id = 'right_wheel'

        # Position: wheel offset from base
        right_wheel_transform.transform.translation.x = 0.15
        right_wheel_transform.transform.translation.y = -0.25
        right_wheel_transform.transform.translation.z = -0.1

        # Rotation: account for wheel rotation around Y axis
        wheel_angle = self.joint_positions['right_wheel_joint']
        cy = math.cos(wheel_angle / 2.0)
        sy = math.sin(wheel_angle / 2.0)
        cp = math.cos(-math.pi / 4)  # Account for initial wheel orientation
        sp = math.sin(-math.pi / 4)

        right_wheel_transform.transform.rotation.w = cy * cp
        right_wheel_transform.transform.rotation.x = cy * sp
        right_wheel_transform.transform.rotation.y = sy * cp
        right_wheel_transform.transform.rotation.z = sy * sp

        transforms.append(right_wheel_transform)

        # Publish all transforms
        for transform in transforms:
            self.tf_broadcaster.sendTransform(transform)


class URDFValidator(Node):
    """
    Validates URDF models for common issues and best practices.
    """

    def __init__(self):
        super().__init__('urdf_validator')

        # Publisher for validation results
        self.validation_pub = self.create_publisher(
            String,
            '/urdf_validation_result',
            10
        )

        # Setup timer for validation
        self.validation_timer = self.create_timer(15.0, self.validate_model)

        self.get_logger().info("URDF Validator initialized")

    def validate_model(self):
        """Validate the robot model for common issues"""
        validation_results = self.perform_validation()

        validation_msg = String()
        validation_msg.data = validation_results
        self.validation_pub.publish(validation_msg)

        self.get_logger().info(f"URDF Validation Results:\n{validation_results}")

    def perform_validation(self):
        """Perform comprehensive URDF validation"""
        results = []
        results.append("URDF Validation Report")
        results.append("=" * 30)

        # Simulated validation checks
        issues = self.check_model_issues()

        if issues:
            results.append("Issues Found:")
            for issue in issues:
                results.append(f"  - {issue}")
        else:
            results.append("No issues found. Model appears valid.")

        results.append("")
        results.append("Validation completed successfully.")

        return "\n".join(results)

    def check_model_issues(self):
        """Check for common URDF issues"""
        issues = []

        # This would normally parse a real URDF file
        # For demonstration, we'll simulate some common checks

        # Check 1: Mass values
        if 0.0 in [10.0, 0.5, 0.5]:  # Simulated masses
            issues.append("Mass value of 0 found in one or more links")

        # Check 2: Inertia values
        # Simulated inertia check
        test_inertia = {'ixx': 0.416, 'iyy': 0.583, 'izz': 0.194, 'ixy': 0, 'ixz': 0, 'iyz': 0}
        if any(val == 0 and key not in ['ixy', 'ixz', 'iyz'] for key, val in test_inertia.items()):
            if test_inertia['ixx'] == 0 or test_inertia['iyy'] == 0 or test_inertia['izz'] == 0:
                issues.append("Principal inertia values contain zero values")

        # Check 3: Joint connections
        # In a real validator, we would check if all joint parent/child links exist

        # Check 4: Kinematic loop detection
        # Check if there are any kinematic loops in the model

        return issues


def main(args=None):
    """Main function to run URDF/XACRO demonstration nodes"""
    rclpy.init(args=args)

    # Create all nodes
    analyzer_node = URDFAnalyzer()
    xacro_node = XacroProcessor()
    state_publisher_node = RobotStatePublisher()
    validator_node = URDFValidator()

    try:
        # Create executor and add all nodes
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(analyzer_node)
        executor.add_node(xacro_node)
        executor.add_node(state_publisher_node)
        executor.add_node(validator_node)

        # Spin all nodes
        executor.spin()
    except KeyboardInterrupt:
        analyzer_node.get_logger().info('Analyzer node interrupted')
        xacro_node.get_logger().info('XACRO node interrupted')
        state_publisher_node.get_logger().info('State publisher node interrupted')
        validator_node.get_logger().info('Validator node interrupted')
    finally:
        analyzer_node.destroy_node()
        xacro_node.destroy_node()
        state_publisher_node.destroy_node()
        validator_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()