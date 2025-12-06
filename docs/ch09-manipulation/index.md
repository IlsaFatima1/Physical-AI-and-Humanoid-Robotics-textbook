# Chapter 9: Manipulation and Control Systems

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the fundamental concepts of robotic manipulation and control
- Implement kinematic models for robotic arms and manipulators
- Design and implement various control strategies for manipulation tasks
- Integrate perception systems with manipulation for object interaction
- Apply grasping and manipulation planning algorithms
- Evaluate manipulation system performance and handle uncertainties
- Design safe and robust manipulation systems for real-world applications

## 9.1 Introduction to Robotic Manipulation

Robotic manipulation refers to the ability of a robot to purposefully control objects in its environment. This encompasses a wide range of tasks from simple pick-and-place operations to complex assembly and interaction tasks. Manipulation systems typically involve robotic arms with end-effectors that can grasp, move, and manipulate objects.

### 9.1.1 Manipulation System Components

A typical robotic manipulation system consists of:

1. **Manipulator**: The mechanical structure (arm) with multiple degrees of freedom
2. **End-effector**: The tool at the end of the arm (gripper, suction cup, etc.)
3. **Actuators**: Motors or other devices that provide motion
4. **Sensors**: Cameras, force sensors, tactile sensors for feedback
5. **Controller**: Software and hardware that coordinates the system
6. **Perception System**: For identifying and localizing objects

### 9.1.2 Manipulation Challenges

Robotic manipulation faces several key challenges:

- **Uncertainty**: Uncertain object poses, friction, and dynamics
- **Complexity**: High-dimensional configuration spaces
- **Dexterity**: Need for fine motor control and precision
- **Safety**: Ensuring safe interaction with objects and humans
- **Real-time Performance**: Fast response for dynamic tasks

## 9.2 Kinematics for Manipulation

### 9.2.1 Forward Kinematics

Forward kinematics calculates the position and orientation of the end-effector given joint angles. For a robotic arm with n joints:

```
T = A₁(θ₁) × A₂(θ₂) × ... × Aₙ(θₙ)
```

Where T is the transformation matrix representing end-effector pose, and Aᵢ(θᵢ) are the transformation matrices for each joint.

```python
import numpy as np
import math

class ForwardKinematics:
    def __init__(self, dh_parameters):
        """
        Initialize with Denavit-Hartenberg parameters
        dh_parameters: list of [a, alpha, d, theta_offset] for each joint
        """
        self.dh_params = dh_parameters

    def dh_transform(self, a, alpha, d, theta):
        """Calculate Denavit-Hartenberg transformation matrix"""
        return np.array([
            [math.cos(theta), -math.sin(theta)*math.cos(alpha), math.sin(theta)*math.sin(alpha), a*math.cos(theta)],
            [math.sin(theta), math.cos(theta)*math.cos(alpha), -math.cos(theta)*math.sin(alpha), a*math.sin(theta)],
            [0, math.sin(alpha), math.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def calculate_pose(self, joint_angles):
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

    def get_position(self, joint_angles):
        """Get end-effector position from joint angles"""
        transform = self.calculate_pose(joint_angles)
        return transform[0:3, 3]  # Extract position vector

    def get_orientation(self, joint_angles):
        """Get end-effector orientation from joint angles"""
        transform = self.calculate_pose(joint_angles)
        return transform[0:3, 0:3]  # Extract rotation matrix
```

### 9.2.2 Inverse Kinematics

Inverse kinematics solves for joint angles given a desired end-effector pose. This is often more challenging than forward kinematics.

```python
class InverseKinematics:
    def __init__(self, dh_parameters, max_iterations=100, tolerance=1e-6):
        self.dh_params = dh_parameters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.forward_kinematics = ForwardKinematics(dh_parameters)

    def jacobian(self, joint_angles):
        """Calculate the Jacobian matrix for the manipulator"""
        n = len(joint_angles)
        jacobian = np.zeros((6, n))  # 6 DoF (3 position, 3 orientation)

        # Calculate forward kinematics for all joint configurations
        current_transform = np.eye(4)
        joint_positions = [np.zeros(3)]  # Position of each joint

        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset
            joint_transform = self.forward_kinematics.dh_transform(a, alpha, d, theta)
            current_transform = current_transform @ joint_transform
            joint_positions.append(current_transform[0:3, 3])

        # End-effector position
        end_effector_pos = current_transform[0:3, 3]

        # Calculate Jacobian columns
        for i in range(n):
            # For revolute joints
            z_axis = current_transform[0:3, 2]  # z-axis of joint frame
            joint_pos = joint_positions[i]

            # Linear velocity component
            jacobian[0:3, i] = np.cross(z_axis, end_effector_pos - joint_pos)
            # Angular velocity component
            jacobian[3:6, i] = z_axis

        return jacobian

    def solve(self, target_pose, initial_guess):
        """Solve inverse kinematics using Jacobian transpose method"""
        current_angles = np.array(initial_guess)

        for iteration in range(self.max_iterations):
            # Calculate current pose
            current_transform = self.forward_kinematics.calculate_pose(current_angles)
            current_pos = current_transform[0:3, 3]
            current_rot = current_transform[0:3, 0:3]

            target_pos = target_pose[0:3, 3]
            target_rot = target_pose[0:3, 0:3]

            # Calculate position error
            pos_error = target_pos - current_pos

            # Calculate orientation error (simplified as angle-axis representation)
            rot_error_matrix = target_rot @ current_rot.T - current_rot @ target_rot.T
            angle_error = np.array([
                rot_error_matrix[2, 1],
                rot_error_matrix[0, 2],
                rot_error_matrix[1, 0]
            ])

            # Combine position and orientation errors
            error = np.concatenate([pos_error, angle_error])

            # Check if error is within tolerance
            if np.linalg.norm(error) < self.tolerance:
                return current_angles

            # Calculate Jacobian
            jacobian = self.jacobian(current_angles)

            # Update joint angles using Jacobian transpose method
            # Note: For better convergence, use pseudo-inverse instead
            delta_angles = np.linalg.pinv(jacobian) @ error * 0.1  # Learning rate
            current_angles += delta_angles

            # Apply joint limits (simplified)
            current_angles = np.clip(current_angles, -np.pi, np.pi)

        # Return best solution found
        return current_angles

    def solve_analytical_2r(self, target_x, target_y, l1, l2):
        """
        Analytical solution for 2R planar manipulator
        l1, l2: link lengths
        """
        # Calculate distance to target
        r = math.sqrt(target_x**2 + target_y**2)

        # Check if target is reachable
        if r > l1 + l2:
            # Target is outside workspace
            target_x = target_x * (l1 + l2) / r
            target_y = target_y * (l1 + l2) / r
            r = l1 + l2
        elif r < abs(l1 - l2):
            # Target is inside workspace
            return None  # No solution

        # Calculate joint angles
        cos_theta2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
        sin_theta2 = math.sqrt(1 - cos_theta2**2)
        theta2 = math.atan2(sin_theta2, cos_theta2)

        k1 = l1 + l2 * cos_theta2
        k2 = l2 * sin_theta2
        theta1 = math.atan2(target_y, target_x) - math.atan2(k2, k1)

        return [theta1, theta2]
```

## 9.3 Control Systems for Manipulation

### 9.3.1 PID Control for Joint Control

PID (Proportional-Integral-Derivative) control is fundamental for precise joint control:

```python
class PIDController:
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
        derivative = (error - self.previous_error) / self.dt
        d_term = self.kd * derivative

        # Store error for next iteration
        self.previous_error = error

        # Calculate output
        output = p_term + i_term + d_term

        return output

class JointController:
    def __init__(self, joint_limits=(-np.pi, np.pi)):
        self.joint_limits = joint_limits
        self.pid_controllers = {}

    def add_joint(self, joint_name, kp, ki, kd):
        """Add a PID controller for a specific joint"""
        self.pid_controllers[joint_name] = PIDController(kp, ki, kd)

    def compute_torques(self, joint_positions, joint_velocities, desired_positions, desired_velocities):
        """Compute torques for all joints"""
        torques = {}

        for joint_name, pid in self.pid_controllers.items():
            if joint_name in joint_positions and joint_name in desired_positions:
                # Compute position error
                pos_error = desired_positions[joint_name] - joint_positions[joint_name]

                # Compute velocity error
                vel_error = 0
                if joint_name in joint_velocities and joint_name in desired_velocities:
                    vel_error = desired_velocities[joint_name] - joint_velocities[joint_name]

                # Use position error for P term, velocity error for D term
                pos_control = pid.kp * pos_error
                vel_control = pid.kd * vel_error

                # Total control effort
                control_effort = pos_control + vel_control

                # Apply joint limits
                control_effort = np.clip(control_effort,
                                       self.joint_limits[0], self.joint_limits[1])

                torques[joint_name] = control_effort

        return torques
```

### 9.3.2 Impedance Control

Impedance control allows the robot to behave like a spring-mass-damper system:

```python
class ImpedanceController:
    def __init__(self, mass, damping, stiffness, dt=0.01):
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness
        self.dt = dt

        # State variables
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)

    def update(self, desired_position, external_force):
        """Update impedance controller state"""
        # Calculate position and velocity errors
        pos_error = desired_position - self.position
        vel_error = -self.velocity  # Assuming desired velocity is 0

        # Calculate impedance force
        spring_force = self.stiffness * pos_error
        damper_force = self.damping * vel_error
        inertia_force = self.mass * (-self.acceleration)  # Opposes acceleration

        # Total force
        total_force = spring_force + damper_force + external_force

        # Calculate acceleration using F = ma
        self.acceleration = total_force / self.mass

        # Update velocity and position using numerical integration
        self.velocity += self.acceleration * self.dt
        self.position += self.velocity * self.dt

        return self.position, self.velocity, self.acceleration

    def set_mass(self, new_mass):
        """Adjust mass parameter"""
        self.mass = new_mass

    def set_damping(self, new_damping):
        """Adjust damping parameter"""
        self.damping = new_damping

    def set_stiffness(self, new_stiffness):
        """Adjust stiffness parameter"""
        self.stiffness = new_stiffness
```

### 9.3.3 Operational Space Control

Operational space control allows direct control in Cartesian space:

```python
class OperationalSpaceController:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.gravity_compensation = True

    def compute_cartesian_control(self, current_joints, desired_pose, desired_twist,
                                  kp_pos=100, ki_pos=0, kd_pos=10,
                                  kp_rot=100, ki_rot=0, kd_rot=10):
        """
        Compute control in operational space
        """
        # Calculate forward kinematics
        current_pose = self.robot_model.forward_kinematics(current_joints)
        current_pos = current_pose[0:3, 3]
        current_rot = current_pose[0:3, 0:3]

        # Calculate Jacobian
        jacobian = self.robot_model.jacobian(current_joints)

        # Calculate pose errors
        pos_error = desired_pose[0:3, 3] - current_pos
        rot_error = self.rotation_error(current_rot, desired_pose[0:3, 0:3])

        # Calculate twist errors
        current_twist = jacobian @ self.robot_model.joint_velocities
        twist_error = desired_twist - current_twist

        # Compute operational space forces
        pos_force = kp_pos * pos_error + kd_pos * twist_error[0:3]
        rot_force = kp_rot * rot_error + kd_rot * twist_error[3:6]

        # Combine forces
        operational_force = np.concatenate([pos_force, rot_force])

        # Transform to joint space using transpose Jacobian
        # For better results, use pseudo-inverse: tau = J.T @ F
        joint_torques = jacobian.T @ operational_force

        # Add gravity compensation if enabled
        if self.gravity_compensation:
            gravity_torques = self.robot_model.gravity_compensation(current_joints)
            joint_torques += gravity_torques

        return joint_torques

    def rotation_error(self, current_rot, desired_rot):
        """Calculate rotation error as axis-angle representation"""
        rot_error_matrix = desired_rot @ current_rot.T
        angle = math.acos(min(max((np.trace(rot_error_matrix) - 1) / 2, -1), 1))

        if abs(angle) < 1e-6:
            return np.zeros(3)

        # Calculate axis of rotation
        axis = np.array([
            rot_error_matrix[2, 1] - rot_error_matrix[1, 2],
            rot_error_matrix[0, 2] - rot_error_matrix[2, 0],
            rot_error_matrix[1, 0] - rot_error_matrix[0, 1]
        ]) / (2 * math.sin(angle))

        return angle * axis
```

## 9.4 Grasping and Grasp Planning

### 9.4.1 Grasp Representation

Grasps are typically represented by the position, orientation, and configuration of the end-effector:

```python
class Grasp:
    def __init__(self, position, orientation, approach_direction, grasp_width=0.05):
        self.position = np.array(position)  # 3D position
        self.orientation = np.array(orientation)  # 3D orientation (e.g., quaternion)
        self.approach_direction = np.array(approach_direction)  # Approach direction
        self.grasp_width = grasp_width  # Required gripper width
        self.quality = 0.0  # Grasp quality metric
        self.forces = []  # Expected forces at contact points

    def to_transformation_matrix(self):
        """Convert grasp to transformation matrix"""
        # Create rotation matrix from orientation
        # This is a simplified version - in practice, you'd use quaternion to matrix conversion
        rotation = self.orientation_to_rotation_matrix(self.orientation)

        # Create transformation matrix
        transform = np.eye(4)
        transform[0:3, 0:3] = rotation
        transform[0:3, 3] = self.position

        return transform

    def orientation_to_rotation_matrix(self, orientation):
        """Convert orientation (quaternion) to rotation matrix"""
        # Assuming orientation is a quaternion [x, y, z, w]
        x, y, z, w = orientation

        # Calculate rotation matrix from quaternion
        rotation = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

        return rotation

    def is_feasible(self, object_properties, robot_properties):
        """Check if grasp is feasible given object and robot properties"""
        # Check if grasp width is appropriate for object
        if self.grasp_width > object_properties['max_width']:
            return False

        # Check if approach direction is valid
        # (e.g., not colliding with object support surface)

        # Check if robot can reach the grasp pose
        try:
            ik_solution = robot_properties['ik_solver'].solve(
                self.to_transformation_matrix(),
                robot_properties['initial_joints']
            )
            if ik_solution is not None:
                return True
        except:
            pass

        return False
```

### 9.4.2 Grasp Planning Algorithms

```python
class GraspPlanner:
    def __init__(self, robot_model, gripper_model):
        self.robot_model = robot_model
        self.gripper_model = gripper_model
        self.approach_directions = [
            [0, 0, 1],   # From above
            [0, 0, -1],  # From below
            [1, 0, 0],   # From front
            [-1, 0, 0],  # From back
            [0, 1, 0],   # From left
            [0, -1, 0]   # From right
        ]

    def plan_grasps(self, object_mesh, object_pose):
        """Plan potential grasps for an object"""
        grasps = []

        # Sample points on the object surface
        surface_points = self.sample_surface_points(object_mesh)

        for point in surface_points:
            # Calculate surface normal at this point
            normal = self.calculate_surface_normal(object_mesh, point)

            # Generate grasps for different approach directions
            for approach in self.approach_directions:
                # Check if approach direction is valid (not opposite to normal)
                if np.dot(approach, normal) > 0.1:  # At least 10% alignment
                    grasp = self.create_grasp(point, normal, approach)
                    if grasp:
                        # Evaluate grasp quality
                        quality = self.evaluate_grasp_quality(grasp, object_mesh)
                        grasp.quality = quality
                        grasps.append(grasp)

        # Sort grasps by quality
        grasps.sort(key=lambda g: g.quality, reverse=True)

        return grasps

    def sample_surface_points(self, object_mesh):
        """Sample points on the object surface"""
        # This is a simplified approach
        # In practice, you'd use more sophisticated sampling methods
        points = []

        # For a simple box, sample corner points and face centers
        # This is just an example - real implementation would depend on object representation
        for x in [-0.1, 0, 0.1]:
            for y in [-0.1, 0, 0.1]:
                for z in [-0.1, 0, 0.1]:
                    if abs(x) == 0.1 or abs(y) == 0.1 or abs(z) == 0.1:
                        points.append([x, y, z])

        return points

    def calculate_surface_normal(self, object_mesh, point):
        """Calculate surface normal at a point on the mesh"""
        # Simplified normal calculation
        # In practice, this would involve mesh processing
        # For now, return a simple normal based on position
        normal = np.array(point)
        if np.linalg.norm(normal) > 0:
            normal = normal / np.linalg.norm(normal)
        else:
            normal = np.array([0, 0, 1])

        return normal

    def create_grasp(self, position, normal, approach_direction):
        """Create a grasp at the specified position and orientation"""
        # Calculate orientation based on normal and approach direction
        approach = np.array(approach_direction)
        approach = approach / np.linalg.norm(approach)

        normal = np.array(normal)
        normal = normal / np.linalg.norm(normal)

        # Calculate the orientation that aligns the gripper with the surface
        # This is a simplified calculation
        z_axis = -approach  # Gripper closing direction
        x_axis = np.cross(normal, z_axis)
        if np.linalg.norm(x_axis) < 0.1:  # Parallel vectors
            x_axis = np.array([1, 0, 0])  # Default x-axis
        else:
            x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Create rotation matrix
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

        # Convert to quaternion (simplified)
        # In practice, use proper quaternion conversion
        quat = self.rotation_matrix_to_quaternion(rotation_matrix)

        # Create grasp
        grasp = Grasp(position, quat, approach_direction)

        return grasp

    def rotation_matrix_to_quaternion(self, rotation_matrix):
        """Convert rotation matrix to quaternion"""
        trace = np.trace(rotation_matrix)

        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2  # S=4*qw
            qw = 0.25 * s
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        else:
            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                s = math.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
                qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                qx = 0.25 * s
                qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                s = math.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
                qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                qy = 0.25 * s
                qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            else:
                s = math.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
                qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                qz = 0.25 * s

        return np.array([qx, qy, qz, qw])

    def evaluate_grasp_quality(self, grasp, object_mesh):
        """Evaluate the quality of a grasp"""
        # This is a simplified quality evaluation
        # In practice, this would involve more complex physics simulation

        quality = 0.0

        # Factors affecting grasp quality:
        # 1. Force closure (ability to resist external forces)
        # 2. Grasp stability
        # 3. Accessibility (can robot reach the grasp?)
        # 4. Object properties (friction, weight, etc.)

        # For now, use a simple heuristic based on approach direction
        # and surface normal alignment
        approach = grasp.approach_direction
        # Calculate surface normal at grasp position (simplified)
        normal = np.array(grasp.orientation[0:3])  # Simplified

        # Prefer grasps where approach direction is aligned with normal
        alignment = abs(np.dot(approach, normal))
        quality += alignment * 0.3

        # Prefer grasps at the top of objects
        if grasp.position[2] > 0.05:  # Above some threshold
            quality += 0.2

        # Prefer center grasps
        center_dist = np.linalg.norm(grasp.position[0:2])
        if center_dist < 0.05:  # Close to center
            quality += 0.2

        # Add random factor for diversity
        quality += np.random.random() * 0.3

        return min(quality, 1.0)  # Clamp to [0, 1]
```

## 9.5 Manipulation Planning

### 9.5.1 Task and Motion Planning

Task and Motion Planning (TAMP) combines high-level task planning with low-level motion planning:

```python
class TaskAndMotionPlanner:
    def __init__(self, robot_model, scene_objects):
        self.robot_model = robot_model
        self.scene_objects = scene_objects
        self.motion_planner = RRTPlanner(robot_model)
        self.task_planner = TaskPlanner()

    def plan_manipulation_task(self, task_description):
        """Plan a manipulation task combining task and motion planning"""
        # Parse task description (e.g., "pick object A and place it at B")
        task_plan = self.task_planner.plan_task(task_description)

        full_plan = []
        for task_step in task_plan:
            motion_plan = self.plan_motion_for_task(task_step)
            if motion_plan:
                full_plan.extend(motion_plan)
            else:
                # Try alternative task plan
                alternative_task_plan = self.task_planner.generate_alternative(task_step)
                if alternative_task_plan:
                    motion_plan = self.plan_motion_for_task(alternative_task_plan)
                    if motion_plan:
                        full_plan.extend(motion_plan)
                    else:
                        return None  # Cannot complete task
                else:
                    return None  # No alternative available

        return full_plan

    def plan_motion_for_task(self, task_step):
        """Plan motion for a specific task step"""
        if task_step['action'] == 'pick':
            return self.plan_pick_motion(task_step)
        elif task_step['action'] == 'place':
            return self.plan_place_motion(task_step)
        elif task_step['action'] == 'move_to':
            return self.plan_navigation_motion(task_step)
        else:
            return None

    def plan_pick_motion(self, task_step):
        """Plan motion for picking an object"""
        object_name = task_step['object']
        object_pose = self.scene_objects[object_name]['pose']

        # Find a feasible grasp for the object
        grasp_planner = GraspPlanner(self.robot_model, self.gripper_model)
        grasps = grasp_planner.plan_grasps(
            self.scene_objects[object_name]['mesh'],
            object_pose
        )

        # For each grasp, plan a path to reach it
        for grasp in grasps:
            # Plan approach path
            approach_pose = self.calculate_approach_pose(grasp)
            approach_path = self.motion_planner.plan_path(
                self.robot_model.get_current_pose(),
                approach_pose
            )

            if approach_path:
                # Plan grasp execution path
                grasp_path = [grasp.to_transformation_matrix()]

                # Plan lift path after grasp
                lift_pose = self.calculate_lift_pose(grasp)
                lift_path = self.motion_planner.plan_path(
                    grasp.to_transformation_matrix(),
                    lift_pose
                )

                if lift_path:
                    return approach_path + grasp_path + lift_path

        return None  # No feasible grasp found

    def calculate_approach_pose(self, grasp, distance=0.1):
        """Calculate approach pose before grasp"""
        grasp_transform = grasp.to_transformation_matrix()
        approach_transform = grasp_transform.copy()

        # Move back along approach direction
        approach_direction = grasp.approach_direction
        approach_transform[0:3, 3] += distance * np.array(approach_direction)

        return approach_transform

    def calculate_lift_pose(self, grasp, height=0.1):
        """Calculate lift pose after grasp"""
        grasp_transform = grasp.to_transformation_matrix()
        lift_transform = grasp_transform.copy()

        # Move up along z-axis
        lift_transform[0:3, 3][2] += height

        return lift_transform

    def plan_place_motion(self, task_step):
        """Plan motion for placing an object"""
        target_pose = task_step['target_pose']

        # Plan a path to the placement location
        place_path = self.motion_planner.plan_path(
            self.robot_model.get_current_pose(),
            target_pose
        )

        if place_path:
            # Add release action at the end
            place_path.append(('release', {}))

        return place_path
```

### 9.5.2 Trajectory Generation and Smoothing

```python
class TrajectoryGenerator:
    def __init__(self):
        self.max_velocity = 1.0
        self.max_acceleration = 0.5
        self.dt = 0.01

    def generate_minimal_jerk_trajectory(self, start_pos, end_pos, duration):
        """Generate a minimal jerk trajectory between two points"""
        t = np.linspace(0, duration, int(duration / self.dt))
        trajectory = []

        # Minimal jerk trajectory coefficients
        a0 = start_pos
        a1 = 0  # Start velocity = 0
        a2 = 0  # Start acceleration = 0
        a3 = 10 * (end_pos - start_pos) / (duration ** 3)
        a4 = -15 * (end_pos - start_pos) / (duration ** 4)
        a5 = 6 * (end_pos - start_pos) / (duration ** 5)

        for ti in t:
            pos = a0 + a1*ti + a2*ti**2 + a3*ti**3 + a4*ti**4 + a5*ti**5
            vel = a1 + 2*a2*ti + 3*a3*ti**2 + 4*a4*ti**3 + 5*a5*ti**4
            acc = 2*a2 + 6*a3*ti + 12*a4*ti**2 + 20*a5*ti**3

            trajectory.append({
                'position': pos,
                'velocity': vel,
                'acceleration': acc,
                'time': ti
            })

        return trajectory

    def smooth_path(self, path, max_deviation=0.05):
        """Smooth a path using cubic spline interpolation"""
        if len(path) < 3:
            return path

        # Convert path to numpy array for processing
        path_array = np.array(path)

        # Use cubic spline to smooth the path
        from scipy.interpolate import CubicSpline

        # Create parameter for interpolation
        t = np.linspace(0, 1, len(path_array))

        # Create spline for each dimension
        cs = CubicSpline(t, path_array)

        # Generate smooth path
        t_new = np.linspace(0, 1, len(path_array) * 3)  # Increase resolution
        smooth_path = cs(t_new)

        return smooth_path.tolist()

    def velocity_smoothing(self, trajectory, max_velocity=None, max_acceleration=None):
        """Apply velocity and acceleration limits to trajectory"""
        if max_velocity is None:
            max_velocity = self.max_velocity
        if max_acceleration is None:
            max_acceleration = self.max_acceleration

        smoothed_trajectory = []
        prev_vel = 0

        for point in trajectory:
            pos = point['position']
            vel = np.clip(point['velocity'], -max_velocity, max_velocity)
            acc = np.clip(point['acceleration'], -max_acceleration, max_acceleration)

            # Limit acceleration based on previous velocity
            max_change = max_acceleration * self.dt
            vel = np.clip(vel, prev_vel - max_change, prev_vel + max_change)

            smoothed_trajectory.append({
                'position': pos,
                'velocity': vel,
                'acceleration': acc,
                'time': point['time']
            })

            prev_vel = vel

        return smoothed_trajectory
```

## 9.6 ROS 2 Integration for Manipulation

### 9.6.1 MoveIt! Integration

MoveIt! is the standard motion planning framework for ROS:

```python
import rclpy
from rclpy.node import Node
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionFK
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
import numpy as np

class ManipulationControllerNode(Node):
    def __init__(self):
        super().__init__('manipulation_controller')

        # Initialize MoveIt! interface
        self.move_group = None
        self.joint_state = None
        self.end_effector_pose = None

        # Setup subscribers and publishers
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        self.command_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10
        )

        # Setup manipulation services
        self.ik_service = self.create_client(
            GetPositionIK, '/compute_ik'
        )

        self.fk_service = self.create_client(
            GetPositionFK, '/compute_fk'
        )

        # Setup timers for control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Manipulation Controller initialized")

    def joint_state_callback(self, msg):
        """Update joint state"""
        self.joint_state = msg

    def move_to_pose(self, target_pose):
        """Move end-effector to target pose using inverse kinematics"""
        if not self.ik_service.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('IK service not available')
            return False

        # Create IK request
        ik_request = GetPositionIK.Request()
        ik_request.ik_request.group_name = "manipulator"  # Your move group name
        ik_request.ik_request.pose_stamped = target_pose
        ik_request.ik_request.timeout.sec = 5
        ik_request.ik_request.avoid_collisions = True

        # Call IK service
        future = self.ik_service.call_async(ik_request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            result = future.result()
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                # Execute trajectory
                joint_trajectory = self.create_trajectory_from_ik_result(result.solution)
                self.execute_trajectory(joint_trajectory)
                return True
            else:
                self.get_logger().error(f'IK failed with error: {result.error_code.val}')
                return False
        else:
            self.get_logger().error('Failed to call IK service')
            return False

    def create_trajectory_from_ik_result(self, ik_solution):
        """Create joint trajectory from IK solution"""
        trajectory = JointTrajectory()
        trajectory.joint_names = ik_solution.joint_state.name
        trajectory.points = []

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = ik_solution.joint_state.position
        point.velocities = [0.0] * len(ik_solution.joint_state.position)
        point.accelerations = [0.0] * len(ik_solution.joint_state.position)
        point.time_from_start.sec = 3  # 3 seconds to reach target
        point.time_from_start.nanosec = 0

        trajectory.points.append(point)
        return trajectory

    def execute_trajectory(self, trajectory):
        """Execute joint trajectory"""
        self.command_pub.publish(trajectory)

    def pick_object(self, object_pose, approach_height=0.1, lift_height=0.1):
        """Execute pick operation"""
        # 1. Move to approach position above object
        approach_pose = PoseStamped()
        approach_pose.pose = object_pose
        approach_pose.pose.position.z += approach_height
        approach_pose.header.frame_id = "base_link"

        success = self.move_to_pose(approach_pose)
        if not success:
            return False

        # 2. Move down to object
        object_pose_stamped = PoseStamped()
        object_pose_stamped.pose = object_pose
        object_pose_stamped.header.frame_id = "base_link"

        success = self.move_to_pose(object_pose_stamped)
        if not success:
            return False

        # 3. Close gripper (publish gripper command)
        # This would depend on your gripper interface
        self.close_gripper()

        # 4. Lift object
        lift_pose = PoseStamped()
        lift_pose.pose = object_pose
        lift_pose.pose.position.z += lift_height
        lift_pose.header.frame_id = "base_link"

        success = self.move_to_pose(lift_pose)
        return success

    def place_object(self, target_pose, place_height=0.1):
        """Execute place operation"""
        # 1. Move to above placement position
        approach_pose = PoseStamped()
        approach_pose.pose = target_pose
        approach_pose.pose.position.z += place_height
        approach_pose.header.frame_id = "base_link"

        success = self.move_to_pose(approach_pose)
        if not success:
            return False

        # 2. Move down to placement position
        place_pose_stamped = PoseStamped()
        place_pose_stamped.pose = target_pose
        place_pose_stamped.header.frame_id = "base_link"

        success = self.move_to_pose(place_pose_stamped)
        if not success:
            return False

        # 3. Open gripper
        self.open_gripper()

        # 4. Lift gripper
        lift_pose = PoseStamped()
        lift_pose.pose = target_pose
        lift_pose.pose.position.z += place_height
        lift_pose.header.frame_id = "base_link"

        success = self.move_to_pose(lift_pose)
        return success

    def close_gripper(self):
        """Close the gripper"""
        # Implementation depends on gripper type
        # This is a placeholder
        self.get_logger().info("Closing gripper")

    def open_gripper(self):
        """Open the gripper"""
        # Implementation depends on gripper type
        # This is a placeholder
        self.get_logger().info("Opening gripper")

    def control_loop(self):
        """Main control loop for manipulation"""
        # This would contain the main manipulation logic
        # For now, it just logs the current state
        if self.joint_state:
            self.get_logger().debug(f"Current joint positions: {self.joint_state.position[:3]}...")

class SimpleGripperController:
    """Simple controller for parallel jaw gripper"""
    def __init__(self, node, gripper_joint_name="gripper_joint"):
        self.node = node
        self.gripper_joint_name = gripper_joint_name
        self.current_width = 0.0

        # Publisher for gripper commands
        self.gripper_pub = node.create_publisher(
            JointTrajectory, '/gripper_controller/joint_trajectory', 10
        )

    def set_gripper_width(self, width):
        """Set gripper width in meters"""
        self.current_width = width

        # Create joint trajectory for gripper
        trajectory = JointTrajectory()
        trajectory.joint_names = [self.gripper_joint_name]
        trajectory.points = []

        point = JointTrajectoryPoint()
        point.positions = [width]  # This might need conversion depending on gripper type
        point.velocities = [0.0]
        point.accelerations = [0.0]
        point.time_from_start.sec = 1  # 1 second to reach position
        point.time_from_start.nanosec = 0

        trajectory.points.append(point)
        self.gripper_pub.publish(trajectory)

    def open(self, width=0.08):
        """Open gripper to specified width"""
        self.set_gripper_width(width)

    def close(self, width=0.01):
        """Close gripper to specified width"""
        self.set_gripper_width(width)

    def grasp(self, object_width):
        """Grasp an object of specified width"""
        # Calculate appropriate gripper width
        gripper_width = max(0.01, object_width - 0.01)  # Leave small gap
        self.close(gripper_width)
```

## 9.7 Perception-Action Integration

### 9.7.1 Visual Servoing

Visual servoing uses visual feedback to control robot motion:

```python
class VisualServoController:
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
        current_feature_pos: (u, v) pixel coordinates of feature in current image
        desired_feature_pos: (u, v) pixel coordinates of desired feature position
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

        # Convert image velocities to Cartesian velocities
        # This requires the interaction matrix (Jacobian of image features)
        # For simplicity, we'll return the image velocity directly
        # In practice, you'd convert this to end-effector velocities
        cartesian_velocity = self.image_to_cartesian_velocity(control_command)

        return cartesian_velocity

    def image_to_cartesian_velocity(self, image_velocity):
        """
        Convert image velocity to Cartesian velocity
        This is a simplified version - full implementation requires interaction matrix
        """
        # Simple approximation: assume constant depth
        # In reality, you'd need to compute the interaction matrix
        z_depth = 1.0  # Assumed depth in meters

        # Extract camera parameters
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Convert image velocity to angular velocity
        # This is a simplified approximation
        angular_velocity = np.zeros(6)  # [vx, vy, vz, wx, wy, wz]

        # Approximate linear velocity components
        angular_velocity[0] = -z_depth / fx * image_velocity[0]  # vx
        angular_velocity[1] = -z_depth / fy * image_velocity[1]  # vy

        return angular_velocity

    def reset(self):
        """Reset error integrals and derivatives"""
        self.error_integral = np.zeros(2)
        self.previous_error = np.zeros(2)

class FeatureTracker:
    """Track visual features for visual servoing"""
    def __init__(self):
        import cv2
        self.detector = cv2.SIFT_create()  # or ORB, AKAZE, etc.
        self.matcher = cv2.BFMatcher()
        self.previous_keypoints = None
        self.previous_descriptors = None

    def track_feature(self, current_image, template_image):
        """Track a feature from template to current image"""
        # Detect keypoints and descriptors
        kp_current, desc_current = self.detector.detectAndCompute(current_image, None)
        kp_template, desc_template = self.detector.detectAndCompute(template_image, None)

        if desc_current is None or desc_template is None:
            return None

        # Match features
        matches = self.matcher.knnMatch(desc_template, desc_current, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) >= 4:
            # Get matched keypoints
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_current[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Calculate homography
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Apply homography to template center to find current position
            h, w = template_image.shape
            template_center = np.float32([[w/2, h/2, 1]]).T
            current_center = homography @ template_center
            current_center = current_center / current_center[2]  # Normalize

            return (int(current_center[0]), int(current_center[1]))

        return None
```

## 9.8 Safety and Robustness

### 9.8.1 Force Control and Compliance

Force control allows robots to be compliant and safe when interacting with the environment:

```python
class ForceController:
    def __init__(self, desired_force, stiffness=1000, damping=10):
        self.desired_force = desired_force
        self.stiffness = stiffness
        self.damping = damping

        # Force PID controller
        self.force_pid = PIDController(kp=1.0, ki=0.1, kd=0.05)

        # Previous values for derivative calculation
        self.previous_force_error = 0
        self.force_integral = 0

    def compute_compliance_motion(self, current_force, current_position, dt=0.01):
        """Compute motion based on force feedback"""
        # Calculate force error
        force_error = self.desired_force - current_force

        # Use PID to calculate position adjustment
        position_adjustment = self.force_pid.compute(force_error, 0)  # 0 is measured value (we want error = 0)

        # Calculate desired position
        desired_position = current_position + position_adjustment * dt

        return desired_position

    def impedance_control(self, desired_position, current_position, current_velocity,
                         external_force, dt=0.01):
        """Implement impedance control"""
        # Calculate position and velocity errors
        pos_error = desired_position - current_position
        vel_error = -current_velocity  # Assuming desired velocity is 0

        # Calculate impedance forces
        spring_force = self.stiffness * pos_error
        damper_force = self.damping * vel_error

        # Total force
        total_force = spring_force + damper_force + external_force

        # Calculate acceleration (F = ma, so a = F/m)
        # For simplicity, assume unit mass
        acceleration = total_force

        # Update velocity and position
        new_velocity = current_velocity + acceleration * dt
        new_position = current_position + new_velocity * dt

        return new_position, new_velocity

class SafetyMonitor:
    """Monitor manipulation for safety"""
    def __init__(self):
        self.max_force_threshold = 50.0  # Newtons
        self.max_velocity_threshold = 1.0  # m/s
        self.collision_threshold = 0.05  # meters to obstacles
        self.emergency_stop = False

    def check_safety(self, current_forces, current_velocities, proximity_sensors):
        """Check if current state is safe"""
        # Check force limits
        if any(abs(f) > self.max_force_threshold for f in current_forces):
            self.emergency_stop = True
            return False, "Force limit exceeded"

        # Check velocity limits
        if any(abs(v) > self.max_velocity_threshold for v in current_velocities):
            self.emergency_stop = True
            return False, "Velocity limit exceeded"

        # Check proximity to obstacles
        if any(dist < self.collision_threshold for dist in proximity_sensors):
            self.emergency_stop = True
            return False, "Collision imminent"

        return True, "Safe"

    def reset_safety(self):
        """Reset safety monitor"""
        self.emergency_stop = False
```

## 9.9 Integration with Physical AI Systems

Manipulation systems in Physical AI must seamlessly integrate perception, planning, and control:

### 9.9.1 Perception-Action Loop for Manipulation

```python
class IntegratedManipulationSystem:
    def __init__(self):
        # Initialize all components
        self.perception_module = PerceptionModule()
        self.grasp_planner = GraspPlanner(None, None)
        self.motion_planner = TaskAndMotionPlanner(None, {})
        self.controller = ManipulationControllerNode()
        self.visual_servo = VisualServoController(None)
        self.safety_monitor = SafetyMonitor()

        # Task queue
        self.task_queue = []
        self.current_task = None

    def execute_manipulation_task(self, task_description):
        """Execute a complete manipulation task"""
        # 1. Perceive the environment
        objects = self.perception_module.detect_objects()

        # 2. Plan grasps for target object
        target_object = self.find_target_object(task_description, objects)
        if not target_object:
            return False, "Target object not found"

        grasps = self.grasp_planner.plan_grasps(
            target_object['mesh'],
            target_object['pose']
        )

        # 3. Execute manipulation with best grasp
        for grasp in grasps:
            success = self.execute_grasp(grasp, target_object)
            if success:
                return True, "Successfully executed manipulation task"

        return False, "No feasible grasp found"

    def execute_grasp(self, grasp, target_object):
        """Execute a specific grasp"""
        # Check safety before execution
        is_safe, reason = self.safety_monitor.check_safety(
            [0, 0, 0], [0, 0, 0], [1.0, 1.0, 1.0]
        )

        if not is_safe:
            return False

        # Move to approach pose
        approach_pose = self.calculate_approach_pose(grasp)
        success = self.controller.move_to_pose(approach_pose)
        if not success:
            return False

        # Use visual servoing to fine-tune position if needed
        if self.needs_fine_alignment(target_object):
            self.visual_servo_alignment(target_object, grasp)

        # Execute grasp
        self.execute_approach_and_grasp(grasp)

        # Lift object
        lift_pose = self.calculate_lift_pose(grasp)
        success = self.controller.move_to_pose(lift_pose)

        return success

    def calculate_approach_pose(self, grasp, distance=0.1):
        """Calculate approach pose before grasp"""
        grasp_transform = grasp.to_transformation_matrix()
        approach_transform = grasp_transform.copy()

        # Move back along approach direction
        approach_direction = grasp.approach_direction
        approach_transform[0:3, 3] += distance * np.array(approach_direction)

        return approach_transform

    def calculate_lift_pose(self, grasp, height=0.1):
        """Calculate lift pose after grasp"""
        grasp_transform = grasp.to_transformation_matrix()
        lift_transform = grasp_transform.copy()

        # Move up along z-axis
        lift_transform[0:3, 3][2] += height

        return lift_transform

    def needs_fine_alignment(self, target_object):
        """Determine if fine alignment is needed"""
        # This would be based on object properties and task requirements
        return target_object.get('precision_required', False)

    def visual_servo_alignment(self, target_object, grasp):
        """Use visual servoing for fine alignment"""
        # Get current camera image
        current_image = self.perception_module.get_current_image()

        # Track target feature
        target_feature_pos = self.get_target_feature_position(target_object)
        current_feature_pos = self.visual_servo.track_feature(
            current_image, target_feature_pos
        )

        # Compute control to align features
        if current_feature_pos is not None:
            velocity = self.visual_servo.compute_control(
                current_feature_pos, target_feature_pos
            )

            # Apply small adjustment to approach pose
            self.apply_velocity_correction(velocity)

    def execute_approach_and_grasp(self, grasp):
        """Execute the approach and grasp motion"""
        # Move to grasp pose
        grasp_pose = grasp.to_transformation_matrix()
        self.controller.move_to_pose(grasp_pose)

        # Close gripper
        self.controller.close_gripper()

    def find_target_object(self, task_description, detected_objects):
        """Find the target object based on task description"""
        for obj in detected_objects:
            if task_description['object_name'] in obj['name']:
                return obj
        return None

    def apply_velocity_correction(self, velocity):
        """Apply small velocity corrections during manipulation"""
        # This would send small velocity commands to adjust position
        # Implementation depends on the robot's control interface
        pass
```

## Summary

Robotic manipulation and control systems form the core of dexterous robot capabilities. The integration of kinematics, dynamics, control theory, and perception enables robots to interact with their environment in meaningful ways. Modern manipulation systems leverage advanced control techniques like operational space control and impedance control to achieve safe and robust interaction. The combination of perception, planning, and control in a unified framework allows robots to perform complex manipulation tasks in unstructured environments. As Physical AI systems continue to evolve, manipulation capabilities will become increasingly sophisticated, enabling robots to perform human-like dexterous tasks safely and efficiently.