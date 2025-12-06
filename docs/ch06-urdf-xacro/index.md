# Chapter 6: URDF and XACRO for Robot Modeling

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the structure and components of Unified Robot Description Format (URDF)
- Create complex robot models using URDF with proper kinematic chains
- Use XACRO (XML Macros) to simplify and parameterize robot descriptions
- Integrate URDF models with ROS 2 for visualization and simulation
- Apply best practices for robot modeling and maintainability
- Validate URDF models and debug common issues

## 6.1 Introduction to URDF (Unified Robot Description Format)

Unified Robot Description Format (URDF) is an XML-based format used to describe robots in ROS. It defines the physical and kinematic properties of a robot, including its links, joints, and other associated properties like visual and collision representations.

### 6.1.1 Why Robot Description is Important

Robot description is crucial for:
- **Simulation**: Creating accurate simulation models
- **Visualization**: Displaying robots in RViz and other tools
- **Kinematic Analysis**: Computing forward and inverse kinematics
- **Collision Detection**: Determining when parts of a robot collide
- **Motion Planning**: Planning paths while avoiding self-collision

### 6.1.2 URDF Core Concepts

URDF models consist of two fundamental elements:
- **Links**: Rigid bodies with physical properties (mass, inertia, visual representation)
- **Joints**: Connections between links that constrain their relative motion

## 6.2 URDF Structure and Components

### 6.2.1 Basic URDF Document Structure

```xml
<?xml version="1.0"?>
<robot name="my_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Links definition -->
  <link name="base_link">
    <!-- Link properties -->
  </link>

  <!-- Joints definition -->
  <joint name="joint_name" type="joint_type">
    <parent link="parent_link_name"/>
    <child link="child_link_name"/>
  </joint>

  <!-- Other elements (materials, transmissions, etc.) -->
</robot>
```

### 6.2.2 Link Definition

A link represents a rigid body in the robot and contains several sub-elements:

```xml
<link name="link_name">
  <!-- Inertial properties -->
  <inertial>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
  </inertial>

  <!-- Visual properties -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.5 0.5 0.2"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 1 1"/>
    </material>
  </visual>

  <!-- Collision properties -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.5 0.5 0.2"/>
    </geometry>
  </collision>
</link>
```

### 6.2.3 Joint Definition

Joints define the connection between links and their allowed motion:

```xml
<joint name="joint_name" type="joint_type">
  <parent link="parent_link_name"/>
  <child link="child_link_name"/>
  <origin xyz="0.1 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

Joint types include:
- **revolute**: Rotational joint with limited range
- **continuous**: Rotational joint without limits
- **prismatic**: Linear sliding joint with limits
- **fixed**: No relative motion between links
- **floating**: 6 DOF with no constraints
- **planar**: Motion on a plane

## 6.3 Advanced URDF Features

### 6.3.1 Materials and Colors

Materials can be defined separately and referenced by links:

```xml
<material name="red">
  <color rgba="1 0 0 1"/>
</material>

<material name="blue">
  <color rgba="0 0 1 1"/>
</material>

<link name="example_link">
  <visual>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
    <material name="red"/>
  </visual>
</link>
```

### 6.3.2 Transmissions

Transmissions define how joints connect to actuators:

```xml
<transmission name="wheel_transmission">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="wheel_joint">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </joint>
  <actuator name="wheel_motor">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### 6.3.3 Gazebo-Specific Extensions

URDF can include Gazebo-specific elements for simulation:

```xml
<gazebo reference="link_name">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
</gazebo>

<gazebo>
  <plugin name="differential_drive" filename="libgazebo_ros_diff_drive.so">
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    <wheel_separation>0.4</wheel_separation>
    <wheel_diameter>0.2</wheel_diameter>
  </plugin>
</gazebo>
```

## 6.4 Introduction to XACRO

XACRO (XML Macros) is a macro language for XML that extends URDF with features like variables, mathematical expressions, and macros. It makes complex robot descriptions more manageable and reusable.

### 6.4.1 XACRO Declaration

To use XACRO, declare it in the robot tag:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="robot_with_xacro">
  <!-- XACRO content -->
</robot>
```

### 6.4.2 XACRO Properties (Variables)

Define reusable values using properties:

```xml
<xacro:property name="M_PI" value="3.1415926535897931" />
<xacro:property name="wheel_radius" value="0.05" />
<xacro:property name="wheel_width" value="0.02" />
<xacro:property name="base_length" value="0.5" />
<xacro:property name="base_width" value="0.4" />
<xacro:property name="base_height" value="0.2" />
```

### 6.4.3 Mathematical Expressions

XACRO supports mathematical operations:

```xml
<xacro:property name="wheel_offset_x" value="${base_length/2 - wheel_width/2}" />
<xacro:property name="wheel_offset_y" value="${base_width/2 + wheel_radius}" />
```

## 6.5 Creating Complex Robot Models with XACRO

### 6.5.1 Defining Macros

Macros allow reusing common patterns:

```xml
<xacro:macro name="wheel" params="prefix parent xyz rpy">
  <joint name="${prefix}_wheel_joint" type="continuous">
    <parent link="${parent}"/>
    <child link="${prefix}_wheel"/>
    <origin xyz="${xyz}" rpy="${rpy}"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="${prefix}_wheel">
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
      </geometry>
    </collision>
  </link>
</xacro:macro>
```

### 6.5.2 Complete XACRO Example

Here's a complete example of a differential drive robot using XACRO:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="diff_drive_robot">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="wheel_radius" value="0.05" />
  <xacro:property name="wheel_width" value="0.02" />
  <xacro:property name="base_length" value="0.5" />
  <xacro:property name="base_width" value="0.4" />
  <xacro:property name="base_height" value="0.2" />
  <xacro:property name="wheel_offset_x" value="${base_length/2 - wheel_width/2}" />
  <xacro:property name="wheel_offset_y" value="${base_width/2 + wheel_radius}" />

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
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
      <material name="orange">
        <color rgba="1 0.5 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
    </collision>
  </link>

  <!-- Wheel macro -->
  <xacro:macro name="wheel" params="prefix reflect">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="base_link"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${wheel_offset_x*reflect} 0 -${base_height/2}" rpy="${-M_PI/2} 0 0"/>
      <axis xyz="0 0 1"/>
    </joint>

    <link name="${prefix}_wheel">
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
      </inertial>
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
    </link>
  </xacro:macro>

  <!-- Instantiate wheels -->
  <xacro:wheel prefix="left" reflect="1"/>
  <xacro:wheel prefix="right" reflect="-1"/>

  <!-- Castor wheel -->
  <joint name="caster_joint" type="fixed">
    <parent link="base_link"/>
    <child link="caster_wheel"/>
    <origin xyz="${-base_length/2} 0 -${base_height/2 - 0.01}" rpy="0 0 0"/>
  </joint>

  <link name="caster_wheel">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
    <visual>
      <geometry>
        <sphere radius="0.02"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.02"/>
      </geometry>
    </collision>
  </link>

</robot>
```

## 6.6 URDF Best Practices

### 6.6.1 Proper Inertial Properties

Accurate inertial properties are crucial for simulation:

```xml
<inertial>
  <mass value="1.0"/>
  <inertia
    ixx="0.083" ixy="0" ixz="0"
    iyy="0.083" iyz="0"
    izz="0.166"/>
</inertial>
```

For common shapes, use these formulas:
- Box: `Ixx = m*(h² + d²)/12`, `Iyy = m*(w² + d²)/12`, `Izz = m*(w² + h²)/12`
- Cylinder: `Ixx = Iyy = m*(3*r² + h²)/12`, `Izz = m*r²/2`
- Sphere: `Ixx = Iyy = Izz = 2*m*r²/5`

### 6.6.2 Collision vs Visual Geometry

- Use simple shapes for collision geometry (boxes, cylinders, spheres)
- Use detailed meshes for visual geometry
- Ensure collision geometry completely contains visual geometry

### 6.6.3 Proper Origin and Frame Definitions

- Use consistent coordinate frames (typically X-forward, Y-left, Z-up)
- Define origins relative to the parent link's coordinate frame
- Use proper rotation conventions (roll-pitch-yaw)

## 6.7 Integrating URDF with ROS 2

### 6.7.1 Robot State Publisher

The robot_state_publisher node reads the URDF and publishes TF transforms:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
import math

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Read URDF from parameter server or file
        self.declare_parameter('robot_description', '')

        # Subscribe to joint states
        self.joint_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Setup TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

    def joint_state_callback(self, msg):
        """Process joint state messages and publish transforms"""
        # Process each joint and publish corresponding transform
        for i, joint_name in enumerate(msg.name):
            # Create and publish transform for each joint
            pass
```

### 6.7.2 Launch Configuration

Create a launch file to start the robot state publisher with URDF:

```python
# launch/robot_state_publisher.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    urdf_package = DeclareLaunchArgument(
        'urdf_package',
        default_value='my_robot_description',
        description='Package containing robot URDF files'
    )

    urdf_file = DeclareLaunchArgument(
        'urdf_file',
        default_value='robot.urdf.xacro',
        description='URDF file name'
    )

    # Find URDF file
    robot_description_path = PathJoinSubstitution([
        FindPackageShare(LaunchConfiguration('urdf_package')),
        'urdf',
        LaunchConfiguration('urdf_file')
    ])

    # Robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': PathJoinSubstitution([
                FindPackageShare(LaunchConfiguration('urdf_package')),
                'urdf',
                LaunchConfiguration('urdf_file')
            ])
        }]
    )

    return LaunchDescription([
        urdf_package,
        urdf_file,
        robot_state_publisher
    ])
```

## 6.8 Validation and Debugging

### 6.8.1 URDF Validation Tools

ROS provides tools for validating URDF models:

```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# Parse XACRO to URDF
xacro input_file.xacro > output_file.urdf

# Visualize in RViz
ros2 launch rviz2 rviz2
```

### 6.8.2 Common Issues and Solutions

1. **Self-collision**: Check joint limits and ensure proper spacing
2. **Kinematic errors**: Verify joint types and axis directions
3. **Inertial issues**: Use proper mass and inertia values
4. **Visual artifacts**: Check geometry definitions and materials

### 6.8.3 Debugging Techniques

```python
# Python script to check URDF model
import xml.etree.ElementTree as ET

def validate_urdf(urdf_path):
    """Validate basic URDF structure"""
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Check for required elements
        links = root.findall('link')
        joints = root.findall('joint')

        print(f"URDF contains {len(links)} links and {len(joints)} joints")

        # Check for proper joint connections
        link_names = [link.get('name') for link in links]
        for joint in joints:
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')

            if parent not in link_names:
                print(f"Error: Joint {joint.get('name')} references non-existent parent link {parent}")
            if child not in link_names:
                print(f"Error: Joint {joint.get('name')} references non-existent child link {child}")

    except ET.ParseError as e:
        print(f"XML Parse Error: {e}")
    except Exception as e:
        print(f"Error validating URDF: {e}")
```

## 6.9 Advanced XACRO Features

### 6.9.1 Conditional Statements

XACRO supports conditional processing:

```xml
<xacro:property name="use_gpu" value="true" />

<xacro:if value="$(arg use_gpu)">
  <gazebo reference="sensor_link">
    <sensor type="gpu_ray" name="gpu_laser">
      <!-- GPU-accelerated sensor configuration -->
    </sensor>
  </gazebo>
</xacro:if>

<xacro:unless value="$(arg use_gpu)">
  <gazebo reference="sensor_link">
    <sensor type="ray" name="laser">
      <!-- CPU-based sensor configuration -->
    </sensor>
  </gazebo>
</xacro:unless>
```

### 6.9.2 Including Other XACRO Files

Break down complex models into modular files:

```xml
<!-- Include common components -->
<xacro:include filename="$(find my_robot_description)/urdf/materials.xacro" />
<xacro:include filename="$(find my_robot_description)/urdf/wheel.xacro" />
<xacro:include filename="$(find my_robot_description)/urdf/sensors.xacro" />

<!-- Use included components -->
<xacro:wheel prefix="front_left"/>
<xacro:wheel prefix="front_right"/>
```

## 6.10 Integration with Physical AI Systems

URDF models play a crucial role in Physical AI systems by providing the geometric and kinematic foundation for perception, planning, and control algorithms.

### 6.10.1 Perception Integration

- Accurate collision geometry enables proper collision detection
- Visual models provide realistic rendering for simulation
- Proper inertial properties enable realistic physics simulation

### 6.10.2 Planning and Control

- Kinematic models enable forward and inverse kinematics
- Joint limits and types inform motion planning algorithms
- Mass properties affect dynamic planning and control

## Summary

URDF and XACRO are fundamental tools for robot modeling in ROS 2. URDF provides the basic structure for describing robot geometry, kinematics, and dynamics, while XACRO extends this with powerful macro capabilities that make complex models more manageable. Proper use of these tools enables accurate simulation, visualization, and control of robotic systems. The combination of clear structure, parameterization capabilities, and integration with ROS 2 tools makes URDF and XACRO essential for developing sophisticated Physical AI systems.