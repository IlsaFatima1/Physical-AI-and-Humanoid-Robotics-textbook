# Chapter 2 Quiz: Fundamentals of Robotics and AI Integration

## Multiple Choice Questions

**Question 1**: Which of the following is NOT a fundamental characteristic of robots?
A) Sensing
B) Processing
C) Actuation
D) Consciousness

**Correct Answer**: D
**Explanation**: Consciousness is not a fundamental characteristic of robots. The basic characteristics are sensing (perceiving environment), processing (interpreting data), and actuation (performing actions).

**Question 2**: What does the perception-action cycle in AI-robotics integration include?
A) Perception, Reasoning, Action, Learning
B) Sensing, Planning, Execution
C) Input, Processing, Output
D) Detection, Classification, Response

**Correct Answer**: A
**Explanation**: The perception-action cycle includes Perception (sensing and interpreting the environment), Reasoning (processing information and making decisions), Action (executing physical actions), and Learning (updating models based on outcomes).

**Question 3**: Which type of learning is most appropriate for a robot learning to navigate through trial and error with rewards?
A) Supervised Learning
B) Unsupervised Learning
C) Reinforcement Learning
D) Deep Learning

**Correct Answer**: C
**Explanation**: Reinforcement Learning is specifically designed for learning through trial and error with rewards, making it ideal for robot navigation tasks.

**Question 4**: What is the difference between forward and inverse kinematics?
A) Forward is faster than inverse
B) Forward calculates end-effector position from joint angles; inverse calculates joint angles for desired end-effector position
C) Forward is for mobile robots; inverse is for manipulators
D) Forward uses sensors; inverse uses encoders

**Correct Answer**: B
**Explanation**: Forward kinematics calculates the end-effector position based on joint angles, while inverse kinematics calculates the required joint angles to achieve a desired end-effector position.

**Question 5**: Which control architecture uses layered behaviors where higher layers can subsume lower layers?
A) Deliberative Architecture
B) Reactive Architecture
C) Subsumption Architecture
D) Hybrid Architecture

**Correct Answer**: C
**Explanation**: Subsumption Architecture is characterized by layered behaviors where higher layers can override lower layers when needed.

**Question 6**: What is the main challenge in integrating AI with robotics related to time constraints?
A) AI algorithms must operate within robot time constraints
B) Robots operate too slowly for AI
C) Time is not important in robotics
D) AI algorithms are too fast for robots

**Correct Answer**: A
**Explanation**: One of the key challenges in AI-robotics integration is ensuring that AI algorithms operate within the real-time constraints required by physical robots.

**Question 7**: Which of the following is NOT a type of actuation system?
A) Electric Motors
B) Hydraulic Systems
C) Pneumatic Systems
D) Computer Vision Systems

**Correct Answer**: D
**Explanation**: Computer Vision Systems are sensing systems, not actuation systems. Actuation systems provide the power for robot movement.

## Practical Application Questions

**Question 8**: You are designing a robot that needs to pick up objects of various shapes and sizes. Describe the sensing, processing, and actuation components you would need, and explain how they would work together in the perception-action cycle.

**Expected Answer**:
- Sensing: Cameras for object recognition and depth sensors for size determination, force/torque sensors for grip feedback
- Processing: Computer vision algorithms to identify objects and their properties, planning algorithms to determine grasp strategy
- Actuation: Robotic arm with dexterous end effector, joint motors for positioning
- Perception-Action Cycle: Sense object properties → Process information and plan grasp → Actuate to pick up object → Sense success/failure → Learn for future attempts

**Question 9**: A robot is navigating an unknown environment. Identify the challenges related to real-time requirements, uncertainty management, and safety that the robot would face, and suggest approaches to address each.

**Expected Answer**:
- Real-time requirements: Robot must make navigation decisions quickly to avoid obstacles; addressed with efficient algorithms and appropriate hardware
- Uncertainty management: Sensors provide noisy data about environment; addressed with sensor fusion and probabilistic approaches
- Safety: Robot must avoid collisions and operate safely; addressed with collision detection, emergency stops, and safe trajectory planning

## Code Analysis Questions

**Question 10**: Consider a simple robot controller that implements a feedback loop. Analyze the following pseudocode:

```
while robot_is_operational:
    sensor_data = read_sensors()
    processed_data = process_sensor_data(sensor_data)
    control_commands = calculate_control_commands(processed_data, goal)
    send_commands_to_actuators(control_commands)
    wait(0.01)  # 10ms delay
```

What type of control architecture does this represent? What are the potential issues with the fixed delay, and how might you improve it?

**Expected Answer**:
This represents a basic closed-loop control architecture. Potential issues with the fixed delay include: not being responsive to critical events that require immediate action, potential timing issues if processing takes longer than the delay, and inefficient use of computational resources. Improvements could include: using interrupt-based processing for critical events, implementing variable timing based on task requirements, and using real-time scheduling.

## Conceptual Understanding Questions

**Question 11**: Explain the difference between deliberative and reactive robotic systems. Provide examples of tasks where each approach would be most appropriate and justify your reasoning.

**Expected Answer**:
- Deliberative systems plan actions based on reasoning about future states; appropriate for tasks requiring complex planning like route planning in known environments
- Reactive systems respond directly to stimuli without planning; appropriate for tasks requiring quick responses like obstacle avoidance
- Deliberative is better for tasks with predictable environments and complex goals; reactive is better for dynamic environments requiring immediate responses

**Question 12**: Discuss the safety and ethical considerations that arise as robots become more autonomous. How do these considerations affect the design and deployment of AI-robotics systems?

**Expected Answer**:
Safety considerations include physical safety (preventing harm), operational safety (reliable operation), cybersecurity (protection from unauthorized access), and fail-safe mechanisms. Ethical considerations include privacy (protecting personal information), autonomy (balancing robot autonomy with human control), job displacement (impact on employment), and accountability (responsibility for robot actions). These considerations affect design by requiring safety mechanisms, ethical guidelines, and regulatory compliance, and affect deployment by requiring safety testing, ethical review, and ongoing monitoring.