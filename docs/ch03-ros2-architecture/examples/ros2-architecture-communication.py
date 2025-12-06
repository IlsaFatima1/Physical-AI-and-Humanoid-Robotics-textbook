"""
ROS 2 Architecture and Communication Examples
Chapter 3: ROS 2 Architecture and Communication

This module demonstrates various ROS 2 communication patterns including:
- Topics (publish/subscribe)
- Services (request/reply)
- Actions (goal/feedback/result)
- QoS policy configurations
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from example_interfaces.srv import AddTwoInts
from example_interfaces.action import Fibonacci
from rclpy.action import ActionClient, ActionServer
from sensor_msgs.msg import Image
import time
import threading


class ROSStructureDemo(Node):
    """
    Demonstrates core ROS 2 architecture concepts including nodes,
    communication patterns, and QoS policies.
    """

    def __init__(self):
        super().__init__('ros2_structure_demo')

        # 1. Topic Communication with Different QoS Policies
        self.setup_topic_communication()

        # 2. Service Communication
        self.setup_service_communication()

        # 3. Action Communication
        self.setup_action_communication()

        # 4. QoS Policy Demonstrations
        self.setup_qos_demonstrations()

        self.get_logger().info("ROS 2 Structure Demo initialized")

    def setup_topic_communication(self):
        """Set up publisher and subscriber with different QoS profiles"""

        # Reliable communication for critical commands
        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # Best effort for sensor data
        best_effort_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        # Publishers
        self.critical_cmd_publisher = self.create_publisher(
            String, 'critical_commands', reliable_qos)
        self.sensor_data_publisher = self.create_publisher(
            String, 'sensor_data', best_effort_qos)

        # Subscribers
        self.critical_cmd_subscriber = self.create_subscription(
            String, 'critical_commands',
            self.critical_cmd_callback, reliable_qos)
        self.sensor_data_subscriber = self.create_subscription(
            String, 'sensor_data',
            self.sensor_data_callback, best_effort_qos)

        # Timer for publishing messages
        self.counter = 0
        self.topic_timer = self.create_timer(1.0, self.publish_topic_messages)

    def critical_cmd_callback(self, msg):
        """Callback for critical command messages"""
        self.get_logger().info(f'Critical command received: {msg.data}')

    def sensor_data_callback(self, msg):
        """Callback for sensor data messages"""
        self.get_logger().info(f'Sensor data received: {msg.data}')

    def publish_topic_messages(self):
        """Publish example messages to both topics"""
        # Publish critical command
        cmd_msg = String()
        cmd_msg.data = f'Critical Command #{self.counter}'
        self.critical_cmd_publisher.publish(cmd_msg)

        # Publish sensor data
        sensor_msg = String()
        sensor_msg.data = f'Sensor Reading #{self.counter}'
        self.sensor_data_publisher.publish(sensor_msg)

        self.counter += 1

    def setup_service_communication(self):
        """Set up service server and client"""
        # Service server
        self.service_server = self.create_service(
            AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

        # Service client will be created on demand
        self.service_client = None

    def add_two_ints_callback(self, request, response):
        """Service callback that adds two integers"""
        result = request.a + request.b
        response.sum = result
        self.get_logger().info(f'Adding {request.a} + {request.b} = {result}')
        return response

    def setup_action_communication(self):
        """Set up action server and client"""
        # Action server
        self.action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci_action',
            self.execute_fibonacci_action)

    def execute_fibonacci_action(self, goal_handle):
        """Execute the Fibonacci action"""
        self.get_logger().info('Executing Fibonacci action...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Fibonacci action canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            time.sleep(0.5)  # Simulate processing time

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Fibonacci result: {result.sequence}')
        return result

    def setup_qos_demonstrations(self):
        """Set up various QoS policy demonstrations"""

        # Transient-local for configuration parameters
        config_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST
        )

        self.config_publisher = self.create_publisher(
            String, 'robot_config', config_qos)

        # Keep-all for logging (use with caution in production)
        log_qos = QoSProfile(
            depth=0,  # 0 means keep all
            history=HistoryPolicy.KEEP_ALL
        )

        self.log_publisher = self.create_publisher(
            String, 'system_log', log_qos)

    def demonstrate_qos_usage(self):
        """Demonstrate different QoS usage scenarios"""
        # Publish configuration that new nodes should receive
        config_msg = String()
        config_msg.data = 'Max velocity: 1.0 m/s'
        self.config_publisher.publish(config_msg)

        # Publish log entry
        log_msg = String()
        log_msg.data = f'System status at {time.time()}'
        self.log_publisher.publish(log_msg)

    def call_service_async_example(self, a, b):
        """Example of asynchronous service call"""
        if self.service_client is None:
            self.service_client = self.create_client(AddTwoInts, 'add_two_ints')

        while not self.service_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

        request = AddTwoInts.Request()
        request.a = a
        request.b = b

        # Asynchronous call
        future = self.service_client.call_async(request)
        future.add_done_callback(self.service_response_callback)

        return future

    def service_response_callback(self, future):
        """Callback for service response"""
        try:
            response = future.result()
            self.get_logger().info(f'Service result: {response.sum}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')


def main(args=None):
    """Main function to run the ROS 2 structure demo"""
    rclpy.init(args=args)

    # Create the demo node
    demo_node = ROSStructureDemo()

    # Start the action client in a separate thread to demonstrate action usage
    def run_action_client():
        time.sleep(2)  # Wait for server to be ready

        # Create action client
        action_client = ActionClient(demo_node, Fibonacci, 'fibonacci_action')

        # Wait for action server
        action_client.wait_for_server()

        # Send goal
        goal_msg = Fibonacci.Goal()
        goal_msg.order = 5

        send_goal_future = action_client.send_goal_async(
            goal_msg,
            feedback_callback=lambda feedback_msg: demo_node.get_logger().info(
                f'Feedback: {feedback_msg.feedback.sequence}'))

        # Process the goal response
        rclpy.spin_until_future_complete(demo_node, send_goal_future)
        goal_handle = send_goal_future.result()

        if goal_handle.accepted:
            demo_node.get_logger().info('Goal accepted')
            get_result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(demo_node, get_result_future)
            result = get_result_future.result().result
            demo_node.get_logger().info(f'Action result: {result.sequence}')
        else:
            demo_node.get_logger().info('Goal rejected')

    # Start action client in background thread
    action_thread = threading.Thread(target=run_action_client)
    action_thread.start()

    # Run service call example
    def run_service_example():
        time.sleep(1)  # Wait a bit before making service call
        demo_node.call_service_async_example(5, 3)

    service_thread = threading.Thread(target=run_service_example)
    service_thread.start()

    try:
        # Spin the node to handle callbacks
        rclpy.spin(demo_node)
    except KeyboardInterrupt:
        demo_node.get_logger().info('Interrupted by user')
    finally:
        demo_node.destroy_node()
        rclpy.shutdown()
        action_thread.join()
        service_thread.join()


if __name__ == '__main__':
    main()