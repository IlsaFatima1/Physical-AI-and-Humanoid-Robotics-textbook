# Chapter 9 Quiz: Manipulation and Control Systems

## Multiple Choice Questions

1. What does the Jacobian matrix represent in robotic manipulation?
   a) The relationship between joint velocities and end-effector velocities
   b) The mass distribution of the robot links
   c) The transformation between different coordinate frames
   d) The friction coefficients at each joint

2. In inverse kinematics, what is the main challenge when solving for joint angles?
   a) The solution is always unique
   b) Multiple solutions may exist for the same end-effector pose
   c) Forward kinematics must be calculated first
   d) The robot must be stationary

3. What is the primary purpose of impedance control in robotic manipulation?
   a) To maximize robot speed
   b) To make the robot behave like a spring-mass-damper system for safe interaction
   c) To eliminate all external forces
   d) To increase robot stiffness

4. Which control approach directly controls forces rather than positions?
   a) PID control
   b) Operational space control
   c) Impedance control
   d) Force control

5. What is a key advantage of operational space control?
   a) It only works for 2-DOF robots
   b) It allows direct control in Cartesian space
   c) It eliminates the need for inverse kinematics
   d) It always provides the optimal solution

## Practical Application Questions

6. You are designing a manipulation system for a robot that needs to pick up objects of various shapes and sizes. Design a complete system that includes:
   a) A method for generating feasible grasps
   b) A control strategy for approaching and grasping objects
   c) Safety mechanisms to prevent damage to objects and robot
   d) A planning approach that accounts for dynamic environments

7. Implement a hybrid position/force control system for a robot performing a peg-in-hole assembly task. Describe:
   a) How to transition between position control and force control
   b) The control parameters needed for stable insertion
   c) How to detect contact and adjust control accordingly
   d) Methods to handle uncertainty in object positions

8. Design a manipulation system that integrates visual feedback for precise positioning. Consider:
   a) Visual servoing approaches (image-based vs. position-based)
   b) Camera calibration and feature tracking
   c) How to combine visual feedback with force control
   d) Handling occlusions and tracking failures

## Code Analysis Questions

9. Analyze the following inverse kinematics code using the Jacobian transpose method and identify potential issues:
   ```python
   def inverse_kinematics_jacobian_transpose(target_pose, current_joints, robot_model):
       max_iterations = 100
       tolerance = 1e-6

       joint_angles = current_joints.copy()

       for i in range(max_iterations):
           current_pose = robot_model.forward_kinematics(joint_angles)
           error = target_pose - current_pose

           if np.linalg.norm(error) < tolerance:
               return joint_angles

           jacobian = robot_model.jacobian(joint_angles)
           delta_joints = jacobian.T @ error * 0.1  # Learning rate
           joint_angles += delta_joints

       return joint_angles  # Return best solution found
   ```

10. The following force control code has several problems. Identify and correct them:
    ```python
    def force_control(current_force, desired_force, stiffness=1000):
        # Calculate force error
        force_error = current_force - desired_force

        # Simple proportional control
        position_adjustment = stiffness * force_error

        # Return position adjustment (WRONG!)
        return position_adjustment

    def apply_force_control(robot, desired_force):
        current_force = robot.get_force_feedback()
        position_cmd = force_control(current_force, desired_force)

        # Apply position command to achieve desired force
        robot.move_to_position(position_cmd)  # This is incorrect approach
    ```

## Conceptual Questions

11. Explain the differences between position-based and image-based visual servoing. What are the advantages and disadvantages of each approach, and when would you use one over the other?

12. Describe the main challenges in grasp planning and how they are addressed. What factors affect grasp quality, and how do you evaluate whether a grasp will be successful?

13. Compare and contrast different control strategies for robotic manipulation (PID, impedance, operational space, force control). What are the appropriate use cases for each approach?

14. How does uncertainty affect manipulation tasks, and what techniques can be used to handle uncertainty in perception, modeling, and execution? How does this impact planning and control decisions?

---

## Answer Key

### Multiple Choice Answers:
1. a) The relationship between joint velocities and end-effector velocities
2. b) Multiple solutions may exist for the same end-effector pose
3. b) To make the robot behave like a spring-mass-damper system for safe interaction
4. d) Force control
5. b) It allows direct control in Cartesian space

### Practical Application Answers:

6. Complete manipulation system:
   a) Grasp generation: Use geometric analysis of object surfaces, sample contact points, evaluate grasp quality using force closure criteria
   b) Control strategy: Use impedance control for compliant approach, force control for grasp execution
   c) Safety: Implement force/torque limits, velocity limits, collision detection, emergency stops
   d) Planning: Use reactive replanning, dynamic obstacle avoidance, predictive models

7. Hybrid position/force control:
   a) Transition: Use admittance control or switching based on contact detection
   b) Parameters: Low stiffness for insertion phase, appropriate damping for stability
   c) Contact detection: Monitor force/torque sensors, detect sudden changes
   d) Uncertainty: Use visual servoing for initial positioning, adaptive control parameters

8. Visual feedback integration:
   a) Image-based: Control image features directly, good for camera-in-hand; Position-based: Control 3D pose, requires good calibration
   b) Calibration: Intrinsic and extrinsic parameters, feature detection and matching
   c) Combination: Use vision for gross positioning, force control for fine motion
   d) Handling failures: Fallback to position control, tracking reinitialization

### Code Analysis Answers:

9. Issues with the IK code:
   - Uses Jacobian transpose instead of pseudoinverse (poor convergence)
   - No joint limit checking
   - Fixed learning rate may cause instability
   - No singularity detection
   - Improvements: Use `np.linalg.pinv(jacobian)`, add joint limits, adaptive step size:
   ```python
   delta_joints = np.linalg.pinv(jacobian) @ error * 0.1
   joint_angles = np.clip(joint_angles + delta_joints, joint_limits_min, joint_limits_max)
   ```

10. Issues with force control:
   - Force error calculation should be `desired_force - current_force`
   - Directly mapping force error to position is incorrect
   - Should use admittance control or hybrid position/force control
   - Corrections:
   ```python
   def admittance_control(current_force, desired_force, current_pos, stiffness=100):
       force_error = desired_force - current_force
       pos_adjustment = force_error / stiffness
       return current_pos + pos_adjustment
   ```

### Conceptual Answers:

11. Visual servoing differences:
   - Position-based: Controls 3D pose, requires camera calibration, more stable but sensitive to calibration errors
   - Image-based: Controls image features directly, calibration-free but can have singularities
   - Use position-based for precise 3D tasks, image-based for simple feature alignment

12. Grasp planning challenges:
   - Challenges: Object pose uncertainty, contact modeling, force distribution
   - Factors: Contact points, surface normals, friction, object properties
   - Evaluation: Force closure analysis, grasp quality metrics, dynamic simulation

13. Control strategy comparison:
   - PID: Good for joint space control, simple but limited compliance
   - Impedance: Good for environment interaction, safe but complex tuning
   - Operational Space: Direct Cartesian control, good for tasks but computationally intensive
   - Force Control: Direct force regulation, essential for contact tasks

14. Uncertainty handling:
   - Perception uncertainty: Sensor fusion, probabilistic models, robust feature detection
   - Modeling uncertainty: Adaptive control, learning from experience, robust control design
   - Execution uncertainty: Compliance control, reactive behaviors, replanning
   - Impact: More conservative control, safety margins, fallback behaviors