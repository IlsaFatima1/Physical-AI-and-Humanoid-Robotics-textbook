#!/usr/bin/env python3
"""
Chapter 15 Example: Capstone Project - Autonomous Humanoid with VLA

This example demonstrates the complete integration of Vision-Language-Action
capabilities in an autonomous humanoid robot system. It combines perception,
language understanding, and physical action execution as described in the capstone chapter.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState, Imu
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import String, Bool
from builtin_interfaces.msg import Time
import numpy as np
import cv2
from cv_bridge import CvBridge
import time
import threading
from typing import Dict, List, Tuple, Optional
import json


class VisionSystem:
    """
    Vision system component for the humanoid robot.
    Processes camera data to identify objects and understand the environment.
    """
    def __init__(self):
        self.bridge = CvBridge()
        self.object_detector = self._initialize_object_detector()
        self.depth_threshold = 2.0  # meters

    def _initialize_object_detector(self):
        """
        Initialize a simple object detection system.
        In a real implementation, this would use deep learning models.
        """
        # For simulation purposes, we'll create a simple color-based detector
        return {
            'red_lower': np.array([0, 50, 50]),
            'red_upper': np.array([10, 255, 255]),
            'blue_lower': np.array([100, 50, 50]),
            'blue_upper': np.array([130, 255, 255])
        }

    def process_image(self, cv_image: np.ndarray) -> Dict:
        """
        Process an image to identify objects and their properties.
        """
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Detect red objects (example)
        red_mask = cv2.inRange(hsv, self.object_detector['red_lower'], self.object_detector['red_upper'])
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detect blue objects (example)
        blue_mask = cv2.inRange(hsv, self.object_detector['blue_lower'], self.object_detector['blue_upper'])
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []

        # Process red objects
        for contour in red_contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({
                    'type': 'red_object',
                    'center': (x + w//2, y + h//2),
                    'size': (w, h),
                    'area': area,
                    'distance': self._estimate_distance((w, h))
                })

        # Process blue objects
        for contour in blue_contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({
                    'type': 'blue_object',
                    'center': (x + w//2, y + h//2),
                    'size': (w, h),
                    'area': area,
                    'distance': self._estimate_distance((w, h))
                })

        return {
            'objects': objects,
            'image_shape': cv_image.shape,
            'timestamp': time.time()
        }

    def _estimate_distance(self, size: Tuple[int, int]) -> float:
        """
        Simple distance estimation based on object size in image.
        In reality, this would use depth sensors or stereo vision.
        """
        # Simple inverse relationship: larger objects appear closer
        avg_size = (size[0] + size[1]) / 2
        # This is a very simplified model - real systems use calibrated cameras
        return max(0.1, 2.0 - avg_size / 200.0)


class LanguageProcessor:
    """
    Language processing component for the VLA system.
    Interprets natural language commands and converts them to action plans.
    """
    def __init__(self):
        self.command_keywords = {
            'move': ['move', 'go', 'walk', 'navigate'],
            'grasp': ['grasp', 'grab', 'pick', 'take'],
            'place': ['place', 'put', 'set', 'drop'],
            'approach': ['approach', 'come to', 'go to', 'move to'],
            'stop': ['stop', 'halt', 'pause', 'wait']
        }

        self.object_descriptors = {
            'red': ['red', 'reddish', 'crimson', 'scarlet'],
            'blue': ['blue', 'bluish', 'azure', 'navy'],
            'cup': ['cup', 'mug', 'glass', 'container'],
            'box': ['box', 'container', 'package', 'crate']
        }

    def process_command(self, text: str) -> Dict:
        """
        Process natural language command and return structured action plan.
        """
        text_lower = text.lower()

        # Identify action
        action = None
        for action_type, keywords in self.command_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                action = action_type
                break

        # Identify object
        target_object = None
        for obj_type, descriptors in self.object_descriptors.items():
            if any(descriptor in text_lower for descriptor in descriptors):
                target_object = obj_type
                break

        # Extract location if specified
        location = self._extract_location(text_lower)

        # Create action plan
        action_plan = {
            'action': action,
            'target_object': target_object,
            'location': location,
            'original_command': text,
            'confidence': 0.8  # Simulated confidence
        }

        return action_plan

    def _extract_location(self, text: str) -> Optional[str]:
        """
        Extract location information from the command.
        """
        location_keywords = ['kitchen', 'table', 'room', 'desk', 'shelf', 'cabinet']
        for keyword in location_keywords:
            if keyword in text:
                return keyword
        return None


class TaskPlanner:
    """
    Task planning component that converts high-level commands into executable actions.
    """
    def __init__(self):
        self.action_sequences = {
            ('approach', 'object'): ['navigate_to_object', 'stop_at_safe_distance'],
            ('grasp', 'object'): ['approach_object', 'align_gripper', 'grasp_object', 'verify_grasp'],
            ('place', 'object'): ['navigate_to_location', 'align_for_placement', 'release_object', 'verify_placement']
        }

    def plan_task(self, command_plan: Dict, environment_state: Dict) -> List[Dict]:
        """
        Plan a sequence of actions to accomplish the given command.
        """
        if command_plan['action'] is None:
            return [{'type': 'error', 'message': 'Unknown command'}]

        # For simple commands, create a direct action sequence
        if command_plan['action'] == 'move':
            return self._plan_navigation_task(command_plan, environment_state)
        elif command_plan['action'] == 'grasp':
            return self._plan_grasp_task(command_plan, environment_state)
        elif command_plan['action'] == 'approach':
            return self._plan_approach_task(command_plan, environment_state)
        else:
            # Default: navigate to object if target is specified
            if command_plan['target_object']:
                return self._plan_approach_task(command_plan, environment_state)
            else:
                return [{'type': 'error', 'message': 'Insufficient information for task planning'}]

    def _plan_navigation_task(self, command_plan: Dict, env_state: Dict) -> List[Dict]:
        """
        Plan navigation tasks based on command and environment.
        """
        return [
            {'type': 'navigate', 'target': 'location', 'params': {'speed': 0.3}},
            {'type': 'stop', 'params': {}}
        ]

    def _plan_grasp_task(self, command_plan: Dict, env_state: Dict) -> List[Dict]:
        """
        Plan grasping tasks based on command and environment.
        """
        return [
            {'type': 'approach_object', 'target': command_plan['target_object'], 'params': {'distance': 0.3}},
            {'type': 'align_gripper', 'params': {'precision': 'high'}},
            {'type': 'grasp_object', 'target': command_plan['target_object'], 'params': {'force': 0.5}},
            {'type': 'verify_grasp', 'params': {}}
        ]

    def _plan_approach_task(self, command_plan: Dict, env_state: Dict) -> List[Dict]:
        """
        Plan approach tasks based on command and environment.
        """
        return [
            {'type': 'locate_object', 'target': command_plan['target_object'], 'params': {}},
            {'type': 'navigate_to_object', 'target': command_plan['target_object'], 'params': {'distance': 0.5}},
            {'type': 'stop', 'params': {}}
        ]


class MotionController:
    """
    Motion control component that executes planned actions on the robot.
    """
    def __init__(self):
        self.current_joints = [0.0] * 20  # Simulated 20 joints
        self.current_position = [0.0, 0.0, 0.0]  # x, y, theta
        self.is_moving = False

    def execute_action(self, action: Dict, environment_state: Dict) -> Dict:
        """
        Execute a single action and return the result.
        """
        action_type = action['type']

        if action_type == 'navigate':
            result = self._execute_navigation(action, environment_state)
        elif action_type == 'approach_object':
            result = self._execute_approach(action, environment_state)
        elif action_type == 'grasp_object':
            result = self._execute_grasp(action, environment_state)
        elif action_type == 'stop':
            result = self._execute_stop(action)
        elif action_type == 'locate_object':
            result = self._execute_perception(action, environment_state)
        else:
            result = {'success': False, 'message': f'Unknown action type: {action_type}'}

        return result

    def _execute_navigation(self, action: Dict, env_state: Dict) -> Dict:
        """
        Execute navigation action.
        """
        # Simulate movement
        speed = action.get('params', {}).get('speed', 0.3)
        # In simulation, just update position
        self.current_position[0] += speed * 0.1  # Move forward
        self.is_moving = True
        time.sleep(0.1)  # Simulate execution time
        self.is_moving = False

        return {'success': True, 'message': 'Navigation completed', 'position': self.current_position}

    def _execute_approach(self, action: Dict, env_state: Dict) -> Dict:
        """
        Execute approach object action.
        """
        target = action.get('target', 'unknown')
        distance = action.get('params', {}).get('distance', 0.5)

        # Simulate approach to object
        self.is_moving = True
        time.sleep(0.2)  # Simulate approach time
        self.is_moving = False

        return {'success': True, 'message': f'Approached {target} at {distance}m', 'distance': distance}

    def _execute_grasp(self, action: Dict, env_state: Dict) -> Dict:
        """
        Execute grasp object action.
        """
        target = action.get('target', 'unknown')

        # Simulate grasping action
        self.is_moving = True
        time.sleep(0.3)  # Simulate grasp time
        self.is_moving = False

        # Simulate success/failure based on object properties
        success = np.random.random() > 0.2  # 80% success rate

        return {
            'success': success,
            'message': f'Grasp attempt on {target}' + (' succeeded' if success else ' failed'),
            'object_grasped': success
        }

    def _execute_stop(self, action: Dict) -> Dict:
        """
        Execute stop action.
        """
        self.is_moving = False
        return {'success': True, 'message': 'Robot stopped'}

    def _execute_perception(self, action: Dict, env_state: Dict) -> Dict:
        """
        Execute perception action to locate objects.
        """
        target = action.get('target', 'unknown')

        # Simulate perception
        if 'objects' in env_state:
            target_objects = [obj for obj in env_state['objects'] if target in obj['type']]
            if target_objects:
                closest = min(target_objects, key=lambda x: x['distance'])
                return {
                    'success': True,
                    'message': f'Located {target}',
                    'object_info': closest
                }

        return {
            'success': False,
            'message': f'Could not locate {target}',
            'object_info': None
        }


class LearningSystem:
    """
    Learning system that adapts and improves robot performance over time.
    """
    def __init__(self):
        self.experience_buffer = []
        self.performance_metrics = {
            'success_rate': 0.0,
            'average_time': 0.0,
            'most_common_errors': []
        }

    def update(self, command: str, execution_result: Dict):
        """
        Update learning system based on command and execution result.
        """
        experience = {
            'command': command,
            'result': execution_result,
            'timestamp': time.time()
        }

        self.experience_buffer.append(experience)

        # Update performance metrics
        self._update_metrics(execution_result)

    def _update_metrics(self, result: Dict):
        """
        Update performance metrics based on execution result.
        """
        if len(self.experience_buffer) > 0:
            successes = sum(1 for exp in self.experience_buffer if exp['result']['success'])
            self.performance_metrics['success_rate'] = successes / len(self.experience_buffer)

    def get_advice(self, command: str) -> str:
        """
        Provide advice based on past experiences with similar commands.
        """
        # Simple advice based on past performance
        if self.performance_metrics['success_rate'] < 0.7:
            return "Previous attempts had low success rate. Consider checking environment or retrying with different approach."
        else:
            return "Based on past performance, this command should execute successfully."


class CapstoneVLANode(Node):
    """
    Main ROS 2 node that integrates all VLA components for the capstone project.
    """
    def __init__(self):
        super().__init__('capstone_vla_system')

        # Initialize all components
        self.vision_system = VisionSystem()
        self.language_processor = LanguageProcessor()
        self.task_planner = TaskPlanner()
        self.motion_controller = MotionController()
        self.learning_system = LearningSystem()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(Image, 'camera/image_raw', self.image_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, 'joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.imu_callback, 10)
        self.text_cmd_sub = self.create_subscription(String, 'text_commands', self.text_command_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, 'joint_commands', 10)
        self.status_pub = self.create_publisher(String, 'vla_status', 10)

        # Internal state
        self.current_image = None
        self.current_joints = None
        self.current_imu = None
        self.last_command_time = time.time()

        # Timer for main control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Capstone VLA system initialized')

    def image_callback(self, msg: Image):
        """Process incoming camera images"""
        try:
            cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def joint_callback(self, msg: JointState):
        """Process incoming joint state data"""
        self.current_joints = msg

    def imu_callback(self, msg: Imu):
        """Process incoming IMU data for balance and orientation"""
        self.current_imu = msg

    def text_command_callback(self, msg: String):
        """Process incoming text commands"""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Process the command through the VLA pipeline
        threading.Thread(target=self.process_command_async, args=(command,)).start()

    def process_command_async(self, command: str):
        """
        Process a command asynchronously to avoid blocking the main loop.
        """
        try:
            # Step 1: Process language command
            command_plan = self.language_processor.process_command(command)
            self.get_logger().info(f'Command plan: {command_plan}')

            # Step 2: Get current environment state
            env_state = self.get_current_environment_state()

            # Step 3: Plan the task
            action_sequence = self.task_planner.plan_task(command_plan, env_state)
            self.get_logger().info(f'Action sequence: {action_sequence}')

            # Step 4: Execute the plan
            execution_results = []
            for action in action_sequence:
                result = self.motion_controller.execute_action(action, env_state)
                execution_results.append(result)
                self.get_logger().info(f'Action result: {result}')

                # Publish status
                status_msg = String()
                status_msg.data = f"Executing: {action['type']}, Success: {result['success']}"
                self.status_pub.publish(status_msg)

            # Step 5: Learn from the experience
            final_result = execution_results[-1] if execution_results else {'success': False}
            self.learning_system.update(command, final_result)

            # Step 6: Provide feedback
            success_count = sum(1 for r in execution_results if r['success'])
            total_count = len(execution_results)
            feedback = f"Command '{command}' completed. {success_count}/{total_count} actions successful."
            self.get_logger().info(feedback)

        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')
            status_msg = String()
            status_msg.data = f"Error: {str(e)}"
            self.status_pub.publish(status_msg)

    def get_current_environment_state(self) -> Dict:
        """
        Get the current state of the environment based on sensor data.
        """
        env_state = {}

        # Process current image if available
        if self.current_image is not None:
            vision_result = self.vision_system.process_image(self.current_image)
            env_state.update(vision_result)

        # Add joint state if available
        if self.current_joints is not None:
            env_state['joint_state'] = {
                'position': list(self.current_joints.position),
                'velocity': list(self.current_joints.velocity),
                'effort': list(self.current_joints.effort)
            }

        # Add IMU data if available
        if self.current_imu is not None:
            env_state['imu'] = {
                'orientation': [self.current_imu.orientation.x, self.current_imu.orientation.y,
                               self.current_imu.orientation.z, self.current_imu.orientation.w],
                'angular_velocity': [self.current_imu.angular_velocity.x, self.current_imu.angular_velocity.y,
                                   self.current_imu.angular_velocity.z],
                'linear_acceleration': [self.current_imu.linear_acceleration.x, self.current_imu.linear_acceleration.y,
                                      self.current_imu.linear_acceleration.z]
            }

        return env_state

    def control_loop(self):
        """
        Main control loop for the VLA system.
        """
        # This loop runs continuously and maintains system status
        current_time = time.time()

        # Check if we should publish a status update
        if current_time - self.last_command_time > 5.0:  # Every 5 seconds
            status_msg = String()
            status_msg.data = f"VLA System operational. Objects detected: {len(self.get_current_environment_state().get('objects', []))}"
            self.status_pub.publish(status_msg)
            self.last_command_time = current_time


def main(args=None):
    """
    Main function for the capstone VLA system.
    """
    print("=== Chapter 15: Capstone Project - Autonomous Humanoid with VLA ===\n")

    print("This example demonstrates the integration of Vision-Language-Action capabilities")
    print("in an autonomous humanoid robot system as described in the capstone chapter.\n")

    print("System Components:")
    print("• Vision System: Processes camera data to identify objects and understand environment")
    print("• Language Processor: Interprets natural language commands")
    print("• Task Planner: Converts high-level commands to executable action sequences")
    print("• Motion Controller: Executes planned actions on the robot")
    print("• Learning System: Adapts and improves performance over time\n")

    rclpy.init(args=args)

    try:
        node = CapstoneVLANode()

        print("VLA system node created. The system is now ready to receive commands.")
        print("To test the system, send text commands to the 'text_commands' topic.")
        print("Example commands: 'approach the red cup', 'grasp the blue object', 'move forward'\n")

        print("Key features demonstrated:")
        print("• Real-time perception and understanding")
        print("• Natural language command processing")
        print("• Task planning and execution")
        print("• Learning from experience\n")

        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\nVLA system interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()