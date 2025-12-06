"""
Manipulation and Control Systems Examples
Chapter 9: Manipulation and Control Systems

This module demonstrates various manipulation and control concepts including:
- Forward and inverse kinematics
- Control systems (PID, impedance, operational space)
- Grasp planning and execution
- ROS 2 integration for manipulation
- Visual servoing and force control
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import time


class ManipulationSystemNode(Node):
    """
    Comprehensive manipulation system integrating kinematics, control, and planning.
    """
    def __init__(self):
        super().__init__('manipulation_system')

        # Robot state
        self.current_joints = {}
        self.current_pose = None
        self.current_force = np.zeros(6)  # [fx, fy, fz, mx, my, mz]
        self.target_pose = None
        self.is_executing = False

        # Manipulation components
        self.kinematics = RobotKinematics()
        self.controller = JointController()
        self.grasp_planner = GraspPlanner()
        self.trajectory_generator = TrajectoryGenerator()

        # Setup ROS 2 interface
        self.setup_ros_interface()

        # Setup control timer
        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Manipulation System initialized")

    def setup_ros_interface(self):
        """Setup ROS 2 publishers and subscribers"""
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        self.force_torque_sub = self.create_subscription(
            Float64MultiArray, '/wrench', self.force_torque_callback, 10
        )

        self.target_pose_sub = self.create_subscription(
            PoseStamped, '/manipulation_target', self.target_pose_callback, 10
        )

        # Publishers
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10
        )

        self.ee_pose_pub = self.create_publisher(
            PoseStamped, '/end_effector_pose', 10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray, '/manipulation_markers', 10
        )

        # Services for manipulation tasks
        self.pick_service = self.create_service(
            Trigger, '/pick_object', self.handle_pick_request
        )

        self.place_service = self.create_service(
            Trigger, '/place_object', self.handle_place_request
        )

    def joint_state_callback(self, msg):
        """Update current joint state"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joints[name] = msg.position[i]

        # Calculate current end-effector pose using forward kinematics
        joint_positions = [self.current_joints.get(name, 0.0) for name in self.kinematics.joint_names]
        self.current_pose = self.kinematics.forward_kinematics(joint_positions)

        # Publish current pose
        if self.current_pose is not None:
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "base_link"
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.pose.position.x = self.current_pose[0, 3]
            pose_msg.pose.position.y = self.current_pose[1, 3]
            pose_msg.pose.position.z = self.current_pose[2, 3]

            # Convert rotation matrix to quaternion
            rot = R.from_matrix(self.current_pose[0:3, 0:3])
            quat = rot.as_quat()
            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]

            self.ee_pose_pub.publish(pose_msg)

    def force_torque_callback(self, msg):
        """Update force/torque measurements"""
        if len(msg.data) >= 6:
            self.current_force = np.array(msg.data[:6])

    def target_pose_callback(self, msg):
        """Update target pose for manipulation"""
        self.target_pose = np.array([
            [msg.pose.orientation.w, -msg.pose.orientation.z, msg.pose.orientation.y, msg.pose.position.x],
            [msg.pose.orientation.z, msg.pose.orientation.w, -msg.pose.orientation.x, msg.pose.position.y],
            [-msg.pose.orientation.y, msg.pose.orientation.x, msg.pose.orientation.w, msg.pose.position.z],
            [0, 0, 0, 1]
        ])

    def control_loop(self):
        """Main control loop for manipulation"""
        if self.target_pose is not None and not self.is_executing:
            # Plan and execute motion to target pose
            self.execute_motion_to_target()

    def execute_motion_to_target(self):
        """Execute motion to reach target pose"""
        if self.current_pose is None:
            return

        # Calculate required joint angles using inverse kinematics
        target_pos = self.target_pose[0:3, 3]
        target_rot = self.target_pose[0:3, 0:3]

        current_joints = [self.current_joints.get(name, 0.0) for name in self.kinematics.joint_names]

        # Solve inverse kinematics
        desired_joints = self.kinematics.inverse_kinematics(target_pos, target_rot, current_joints)

        if desired_joints is not None:
            # Generate trajectory to reach desired joints
            current_positions = current_joints
            trajectory = self.trajectory_generator.generate_joint_trajectory(
                current_positions, desired_joints
            )

            # Execute trajectory
            self.execute_joint_trajectory(trajectory)

    def execute_joint_trajectory(self, trajectory):
        """Execute a joint trajectory"""
        self.is_executing = True

        # Publish trajectory
        joint_trajectory_msg = JointTrajectory()
        joint_trajectory_msg.joint_names = self.kinematics.joint_names
        joint_trajectory_msg.points = []

        for i, point in enumerate(trajectory):
            traj_point = JointTrajectoryPoint()
            traj_point.positions = point['positions']
            traj_point.velocities = point['velocities']
            traj_point.accelerations = point['accelerations']
            traj_point.time_from_start.sec = int((i + 1) * 0.1)
            traj_point.time_from_start.nanosec = int(((i + 1) * 0.1 - int((i + 1) * 0.1)) * 1e9)

            joint_trajectory_msg.points.append(traj_point)

        self.joint_trajectory_pub.publish(joint_trajectory_msg)
        self.get_logger().info(f"Published trajectory with {len(trajectory)} points")

        # Reset execution flag after a delay
        self.create_timer(trajectory[-1]['time_from_start'].sec + 1.0, self.reset_execution_flag)

    def reset_execution_flag(self):
        """Reset execution flag after motion completion"""
        self.is_executing = False

    def handle_pick_request(self, request, response):
        """Handle pick object request"""
        self.get_logger().info("Received pick request")

        # In a real implementation, this would:
        # 1. Identify object to pick
        # 2. Plan grasp
        # 3. Execute approach, grasp, and lift motions
        # 4. Verify grasp success

        response.success = True
        response.message = "Pick operation initiated"
        return response

    def handle_place_request(self, request, response):
        """Handle place object request"""
        self.get_logger().info("Received place request")

        # In a real implementation, this would:
        # 1. Identify placement location
        # 2. Plan placement motion
        # 3. Execute place and release motions
        # 4. Verify placement success

        response.success = True
        response.message = "Place operation initiated"
        return response


class RobotKinematics:
    """
    Kinematics implementation for a 6-DOF robotic arm.
    """
    def __init__(self):
        # DH parameters for a 6-DOF manipulator (example values)
        # Format: [a, alpha, d, theta_offset] for each joint
        self.dh_params = [
            [0.0, np.pi/2, 0.1, 0.0],    # Joint 1
            [0.5, 0.0, 0.0, 0.0],        # Joint 2
            [0.0, np.pi/2, 0.0, 0.0],    # Joint 3
            [0.0, -np.pi/2, 0.5, 0.0],   # Joint 4
            [0.0, np.pi/2, 0.0, 0.0],    # Joint 5
            [0.0, 0.0, 0.1, 0.0]         # Joint 6
        ]

        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

    def dh_transform(self, a, alpha, d, theta):
        """Calculate Denavit-Hartenberg transformation matrix"""
        return np.array([
            [math.cos(theta), -math.sin(theta)*math.cos(alpha), math.sin(theta)*math.sin(alpha), a*math.cos(theta)],
            [math.sin(theta), math.cos(theta)*math.cos(alpha), -math.cos(theta)*math.sin(alpha), a*math.sin(theta)],
            [0, math.sin(alpha), math.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, joint_angles):
        """Calculate end-effector pose given joint angles"""
        if len(joint_angles) != len(self.dh_params):
            raise ValueError("Number of joint angles must match number of joints")

        # Start with identity transformation
        transform = np.eye(4)

        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset
            joint_transform = self.dh_transform(a, alpha, d, theta)
            transform = transform @ joint_transform

        return transform

    def jacobian(self, joint_angles):
        """Calculate the geometric Jacobian matrix"""
        n = len(joint_angles)
        jacobian = np.zeros((6, n))  # 6 DoF (3 position, 3 orientation)

        # Calculate transformation matrices for all joints
        transforms = [np.eye(4)]
        current_transform = np.eye(4)

        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset
            joint_transform = self.dh_transform(a, alpha, d, theta)
            current_transform = current_transform @ joint_transform
            transforms.append(current_transform)

        # End-effector position
        end_effector_pos = current_transform[0:3, 3]

        # Calculate Jacobian columns
        for i in range(n):
            # Z-axis of joint frame (rotation axis)
            z_axis = transforms[i][0:3, 2]
            # Position of joint i
            joint_pos = transforms[i][0:3, 3]

            # Linear velocity component (from joint to end-effector)
            linear_component = np.cross(z_axis, end_effector_pos - joint_pos)
            # Angular velocity component
            angular_component = z_axis

            jacobian[0:3, i] = linear_component
            jacobian[3:6, i] = angular_component

        return jacobian

    def inverse_kinematics(self, target_pos, target_rot, initial_guess, max_iterations=100, tolerance=1e-6):
        """Solve inverse kinematics using Jacobian pseudoinverse method"""
        current_angles = np.array(initial_guess, dtype=float)

        for iteration in range(max_iterations):
            # Calculate current pose
            current_transform = self.forward_kinematics(current_angles)
            current_pos = current_transform[0:3, 3]
            current_rot = current_transform[0:3, 0:3]

            # Calculate position error
            pos_error = target_pos - current_pos

            # Calculate orientation error (logarithmic map for rotation matrices)
            rot_error_matrix = target_rot @ current_rot.T
            angle = math.acos(min(max((np.trace(rot_error_matrix) - 1) / 2, -1), 1))

            if abs(angle) < 1e-6:
                rot_error = np.zeros(3)
            else:
                # Calculate axis of rotation
                axis = np.array([
                    rot_error_matrix[2, 1] - rot_error_matrix[1, 2],
                    rot_error_matrix[0, 2] - rot_error_matrix[2, 0],
                    rot_error_matrix[1, 0] - rot_error_matrix[0, 1]
                ]) / (2 * math.sin(angle))
                rot_error = angle * axis

            # Combine position and orientation errors
            error = np.concatenate([pos_error, rot_error])

            # Check if error is within tolerance
            if np.linalg.norm(error) < tolerance:
                return current_angles

            # Calculate Jacobian
            jacobian = self.jacobian(current_angles)

            # Update joint angles using Jacobian pseudoinverse
            delta_angles = np.linalg.pinv(jacobian) @ error * 0.1  # Learning rate
            current_angles += delta_angles

            # Apply joint limits
            current_angles = np.clip(current_angles, -np.pi, np.pi)

        # Return best solution found (even if not within tolerance)
        return current_angles


class JointController:
    """
    Joint-level controller with PID control.
    """
    def __init__(self):
        self.joint_controllers = {}
        self.joint_limits = {
            'joint1': (-np.pi, np.pi),
            'joint2': (-np.pi/2, np.pi/2),
            'joint3': (-np.pi, np.pi),
            'joint4': (-np.pi, np.pi),
            'joint5': (-np.pi/2, np.pi/2),
            'joint6': (-np.pi, np.pi)
        }

    def add_joint(self, joint_name, kp=10.0, ki=0.1, kd=0.5):
        """Add a PID controller for a specific joint"""
        self.joint_controllers[joint_name] = {
            'controller': PIDController(kp, ki, kd),
            'setpoint': 0.0,
            'measured_value': 0.0
        }

    def compute_commands(self, desired_positions, current_positions):
        """Compute control commands for all joints"""
        commands = {}

        for joint_name, controller_info in self.joint_controllers.items():
            if joint_name in desired_positions and joint_name in current_positions:
                controller = controller_info['controller']
                setpoint = desired_positions[joint_name]
                measured_value = current_positions[joint_name]

                command = controller.compute(setpoint, measured_value)

                # Apply joint limits
                limits = self.joint_limits.get(joint_name, (-np.pi, np.pi))
                command = np.clip(command, limits[0], limits[1])

                commands[joint_name] = command

        return commands


class PIDController:
    """
    PID controller implementation.
    """
    def __init__(self, kp, ki, kd, dt=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.previous_error = 0
        self.integral = 0

    def compute(self, setpoint, measured_value):
        """Compute control output using PID"""
        error = setpoint - measured_value

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * self.dt
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.previous_error) / self.dt if self.dt > 0 else 0
        d_term = self.kd * derivative

        # Store error for next iteration
        self.previous_error = error

        # Calculate output
        output = p_term + i_term + d_term

        return output


class GraspPlanner:
    """
    Simple grasp planning implementation.
    """
    def __init__(self):
        self.approach_directions = [
            [0, 0, 1],   # From above
            [0, 0, -1],  # From below
            [1, 0, 0],   # From front
            [-1, 0, 0],  # From back
            [0, 1, 0],   # From left
            [0, -1, 0]   # From right
        ]

    def plan_grasps(self, object_info):
        """
        Plan potential grasps for an object.
        object_info: dictionary containing object properties
        """
        grasps = []

        # For simplicity, assume object is a box with known dimensions
        object_dims = object_info.get('dimensions', [0.1, 0.1, 0.1])
        object_pose = object_info.get('pose', np.eye(4))

        # Generate grasps at different positions and orientations
        for i in range(5):  # Generate 5 potential grasps
            # Random position offset on the object surface
            pos_offset = np.array([
                np.random.uniform(-object_dims[0]/2, object_dims[0]/2),
                np.random.uniform(-object_dims[1]/2, object_dims[1]/2),
                object_dims[2]/2  # Top surface
            ])

            # Transform to world coordinates
            grasp_pos = object_pose[0:3, 0:3] @ pos_offset + object_pose[0:3, 3]

            # Random orientation (for parallel jaw gripper)
            roll = np.random.uniform(-np.pi, np.pi)
            pitch = np.random.uniform(-np.pi/4, np.pi/4)
            yaw = np.random.uniform(-np.pi, np.pi)

            # Create rotation matrix
            rot = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

            # Create transformation matrix
            grasp_transform = np.eye(4)
            grasp_transform[0:3, 0:3] = rot
            grasp_transform[0:3, 3] = grasp_pos

            # Calculate grasp quality (simplified)
            quality = self.evaluate_grasp_quality(grasp_transform, object_info)

            grasps.append({
                'pose': grasp_transform,
                'quality': quality,
                'approach_direction': [0, 0, -1]  # Default approach from above
            })

        # Sort grasps by quality
        grasps.sort(key=lambda g: g['quality'], reverse=True)

        return grasps

    def evaluate_grasp_quality(self, grasp_pose, object_info):
        """Evaluate the quality of a grasp"""
        # Simplified quality evaluation
        # In a real implementation, this would involve physics simulation

        quality = 0.0

        # Prefer grasps near the center of the object
        object_center = object_info.get('pose', np.eye(4))[0:3, 3]
        grasp_pos = grasp_pose[0:3, 3]
        dist_to_center = np.linalg.norm(grasp_pos - object_center)

        # Normalize distance (assuming object is about 0.1m in size)
        dist_score = max(0, 1 - dist_to_center / 0.1)
        quality += dist_score * 0.3

        # Prefer stable orientations (z-axis of gripper aligned with object)
        obj_z_axis = object_info.get('pose', np.eye(4))[0:3, 2]
        gripper_z_axis = grasp_pose[0:3, 2]

        alignment = abs(np.dot(obj_z_axis, gripper_z_axis))
        quality += alignment * 0.3

        # Add some randomness for diversity
        quality += np.random.random() * 0.4

        return min(quality, 1.0)


class TrajectoryGenerator:
    """
    Trajectory generation for smooth motion.
    """
    def __init__(self):
        self.max_velocity = 1.0
        self.max_acceleration = 0.5
        self.dt = 0.05

    def generate_joint_trajectory(self, start_positions, end_positions, duration=3.0):
        """Generate a smooth trajectory between start and end joint positions"""
        n_points = int(duration / self.dt)
        n_joints = len(start_positions)

        trajectory = []

        for i in range(n_points + 1):
            t = i / n_points  # Normalized time [0, 1]

            # Use cubic polynomial for smooth interpolation
            # s(t) = start + (end - start) * (3*t^2 - 2*t^3)
            interpolation_factor = 3 * t**2 - 2 * t**3

            positions = []
            velocities = []
            accelerations = []

            for j in range(n_joints):
                start_pos = start_positions[j]
                end_pos = end_positions[j]

                pos = start_pos + (end_pos - start_pos) * interpolation_factor
                vel = (end_pos - start_pos) * (6*t - 6*t**2) / n_points * self.dt
                acc = (end_pos - start_pos) * (6 - 12*t) / (n_points * self.dt)**2

                positions.append(pos)
                velocities.append(vel)
                accelerations.append(acc)

            trajectory_point = {
                'positions': positions,
                'velocities': velocities,
                'accelerations': accelerations,
                'time_from_start': rclpy.time.Duration(seconds=i * self.dt)
            }

            trajectory.append(trajectory_point)

        return trajectory

    def generate_cartesian_trajectory(self, start_pose, end_pose, duration=3.0):
        """Generate a Cartesian trajectory between two poses"""
        n_points = int(duration / self.dt)

        trajectory = []

        for i in range(n_points + 1):
            t = i / n_points  # Normalized time [0, 1]

            # Interpolate position linearly
            start_pos = start_pose[0:3, 3]
            end_pos = end_pose[0:3, 3]
            pos = start_pos + t * (end_pos - start_pos)

            # Interpolate orientation using SLERP (Spherical Linear Interpolation)
            start_rot = R.from_matrix(start_pose[0:3, 0:3])
            end_rot = R.from_matrix(end_pose[0:3, 0:3])

            # Convert to quaternions for interpolation
            start_quat = start_rot.as_quat()
            end_quat = end_rot.as_quat()

            # SLERP interpolation
            # For simplicity, we'll use linear interpolation and re-normalize
            quat = (1 - t) * start_quat + t * end_quat
            quat = quat / np.linalg.norm(quat)  # Normalize

            # Convert back to rotation matrix
            rot_matrix = R.from_quat(quat).as_matrix()

            # Create transformation matrix
            pose = np.eye(4)
            pose[0:3, 0:3] = rot_matrix
            pose[0:3, 3] = pos

            trajectory.append(pose)

        return trajectory


class ForceController:
    """
    Force control implementation for compliant manipulation.
    """
    def __init__(self, desired_force=np.zeros(6), stiffness=np.eye(6)*1000):
        self.desired_force = desired_force
        self.stiffness = stiffness
        self.damping = np.eye(6) * 10  # Damping matrix
        self.mass = np.eye(6) * 1.0    # Mass matrix (simplified)

        # Force PID controller
        self.force_pid = PIDController(kp=1.0, ki=0.1, kd=0.05, dt=0.05)

    def compute_compliance_motion(self, current_force, current_pose, dt=0.05):
        """Compute compliant motion based on force feedback"""
        # Calculate force error
        force_error = self.desired_force - current_force

        # Use force error to calculate position adjustment
        # This is a simplified impedance control approach
        position_adjustment = np.linalg.inv(self.stiffness) @ force_error * dt

        # Calculate new pose
        new_pose = current_pose.copy()
        new_pose[0:3, 3] += position_adjustment[0:3]  # Update position

        # For orientation, we need to be more careful
        # Apply small rotation based on moment error
        angle_axis = position_adjustment[3:6]  # Small angle approximation
        if np.linalg.norm(angle_axis) > 1e-6:
            angle = np.linalg.norm(angle_axis)
            axis = angle_axis / angle
            skew_symmetric = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            rotation_update = np.eye(3) + np.sin(angle) * skew_symmetric + (1 - np.cos(angle)) * (skew_symmetric @ skew_symmetric)
            new_pose[0:3, 0:3] = rotation_update @ current_pose[0:3, 0:3]

        return new_pose

    def admittance_control(self, current_force, desired_pose, current_pose, dt=0.05):
        """Implement admittance control"""
        # Calculate force error
        force_error = self.desired_force - current_force

        # Calculate pose error
        pos_error = desired_pose[0:3, 3] - current_pose[0:3, 3]
        rot_error_matrix = desired_pose[0:3, 0:3] @ current_pose[0:3, 0:3].T
        angle_error = self.rotation_matrix_to_axis_angle(rot_error_matrix)

        total_error = np.concatenate([pos_error, angle_error])

        # Calculate acceleration using admittance model: M*a + B*v + K*x = F
        # For simplicity, assume v = 0 and solve for acceleration
        acceleration = np.linalg.inv(self.mass) @ (force_error + self.stiffness @ total_error)

        # Update velocity and position
        # This is a simplification - in practice you'd integrate properly
        return desired_pose  # Return desired pose with force feedback applied

    def rotation_matrix_to_axis_angle(self, rot_matrix):
        """Convert rotation matrix to axis-angle representation"""
        angle = math.acos(min(max((np.trace(rot_matrix) - 1) / 2, -1), 1))

        if abs(angle) < 1e-6:
            return np.zeros(3)

        # Calculate axis of rotation
        axis = np.array([
            rot_matrix[2, 1] - rot_matrix[1, 2],
            rot_matrix[0, 2] - rot_matrix[2, 0],
            rot_matrix[1, 0] - rot_matrix[0, 1]
        ]) / (2 * math.sin(angle))

        return angle * axis


class VisualServoController:
    """
    Visual servoing controller for camera-based manipulation.
    """
    def __init__(self, camera_matrix, image_width=640, image_height=480):
        self.camera_matrix = camera_matrix
        self.image_width = image_width
        self.image_height = image_height

        # Control gains
        self.kp = 1.0  # Proportional gain
        self.ki = 0.1  # Integral gain
        self.kd = 0.05  # Derivative gain

        # Error history for integral and derivative terms
        self.error_integral = np.zeros(2)
        self.previous_error = np.zeros(2)
        self.dt = 0.05  # Control loop time step

    def compute_control(self, current_feature_pos, desired_feature_pos):
        """
        Compute control velocities based on feature position error
        """
        # Calculate position error in image coordinates
        error = np.array(desired_feature_pos) - np.array(current_feature_pos)

        # Calculate proportional term
        proportional = self.kp * error

        # Calculate integral term
        self.error_integral += error * self.dt
        integral = self.ki * self.error_integral

        # Calculate derivative term
        derivative = self.kd * (error - self.previous_error) / self.dt if self.dt > 0 else np.zeros(2)

        # Total control command
        control_command = proportional + integral + derivative

        # Store current error for next iteration
        self.previous_error = error

        # Convert image velocity to Cartesian velocity
        # This is a simplified version - full implementation requires interaction matrix
        z_depth = 1.0  # Assumed depth in meters

        # Extract camera parameters
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Calculate Cartesian velocity approximation
        cartesian_velocity = np.zeros(6)  # [vx, vy, vz, wx, wy, wz]

        # Approximate linear velocity components
        cartesian_velocity[0] = -z_depth / fx * control_command[0]  # vx
        cartesian_velocity[1] = -z_depth / fy * control_command[1]  # vy

        return cartesian_velocity


def main(args=None):
    """Main function to run the manipulation system"""
    rclpy.init(args=args)

    # Create manipulation system node
    manipulation_node = ManipulationSystemNode()

    try:
        # Spin the node
        rclpy.spin(manipulation_node)
    except KeyboardInterrupt:
        manipulation_node.get_logger().info('Manipulation system interrupted')
    finally:
        manipulation_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()