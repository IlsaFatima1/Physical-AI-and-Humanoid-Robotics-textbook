# Chapter 3: ROS 2 Architecture and Communication

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the fundamental architecture of ROS 2 and how it differs from ROS 1
- Explain the DDS (Data Distribution Service) communication middleware
- Implement ROS 2 nodes, topics, services, and actions
- Configure Quality of Service (QoS) policies for different communication patterns
- Design robust communication patterns for robotic applications
- Use ROS 2 tools for debugging and monitoring communication

## 3.1 Introduction to ROS 2 Architecture

ROS 2 (Robot Operating System 2) represents a significant architectural evolution from ROS 1, designed to address the challenges of deploying robotics applications in production environments. The primary architectural change is the transition from a centralized master-based system to a decentralized, DDS-based communication architecture.

### 3.1.1 Why ROS 2?

ROS 1's centralized master architecture, while effective for research environments, presented several challenges for production robotics:

- **Single Point of Failure**: The master node was critical for all communication; if it failed, the entire system stopped working.
- **Limited Real-Time Support**: The underlying TCPROS/UDPROS transport protocols didn't provide deterministic timing guarantees.
- **Security Concerns**: No built-in authentication, authorization, or encryption mechanisms.
- **Limited Multi-Robot Support**: Complex coordination between multiple robots required external infrastructure.
- **Deployment Complexity**: Difficult to deploy ROS 1 applications in cloud or containerized environments.

ROS 2 addresses these challenges by adopting a decentralized architecture based on the Data Distribution Service (DDS) standard, providing:

- **Fault Tolerance**: No single point of failure; nodes can discover each other dynamically.
- **Real-Time Support**: DDS provides deterministic communication with configurable QoS policies.
- **Security**: Built-in security features including authentication, encryption, and access control.
- **Scalability**: Native support for multi-robot systems and distributed applications.
- **Standardization**: Conformance to the OMG DDS standard, ensuring interoperability.

### 3.1.2 DDS: The Foundation of ROS 2

Data Distribution Service (DDS) is an open standard for real-time, distributed, and data-centric communication. DDS provides a middleware that enables applications to communicate directly with each other without requiring a central broker. This peer-to-peer communication model is fundamental to ROS 2's decentralized architecture.

Key DDS concepts include:

- **Domain**: A communication space where participants can discover and communicate with each other
- **Participant**: An entity that participates in a DDS domain
- **Topic**: A named data channel for publishing and subscribing to data
- **Publisher**: An entity that sends data on a topic
- **Subscriber**: An entity that receives data from a topic
- **DataWriter**: The interface through which a publisher sends data
- **DataReader**: The interface through which a subscriber receives data

## 3.2 Nodes and Processes

In ROS 2, a node is an instance of a computational process that communicates with other nodes. Nodes are the fundamental building blocks of ROS 2 applications, encapsulating functionality and communicating through the ROS 2 communication infrastructure.

### 3.2.1 Node Creation and Lifecycle

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalNode()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3.2.2 Node Names and Namespaces

ROS 2 uses a hierarchical naming system similar to filesystem paths. Node names can be placed in namespaces to organize complex systems:

```
/robot1/sensor_node
/robot1/control_node
/robot2/sensor_node
/robot2/control_node
```

This prevents naming conflicts and allows multiple instances of the same node type to coexist in the system.

## 3.3 Communication Patterns

ROS 2 supports three primary communication patterns: topics (publish/subscribe), services (request/reply), and actions (goal/cancel/feedback/result).

### 3.3.1 Topics - Publish/Subscribe Pattern

Topics provide asynchronous, one-way communication between nodes. Publishers send messages to topics, and subscribers receive messages from topics. This pattern is ideal for streaming data like sensor readings or robot states.

```python
# Publisher example
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        self.timer = self.create_timer(0.5, self.publish_message)
        self.count = 0

    def publish_message(self):
        msg = String()
        msg.data = f'Hello ROS 2: {self.count}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')
        self.count += 1

# Subscriber example
class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: {msg.data}')
```

### 3.3.2 Services - Request/Reply Pattern

Services provide synchronous, bidirectional communication. A client sends a request to a service server, which processes the request and returns a response. This pattern is suitable for operations that have a clear start and end, like setting parameters or executing commands.

```python
# Service server example
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class ServiceServer(Node):
    def __init__(self):
        super().__init__('add_two_ints_server')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {response.sum}')
        return response

# Service client example
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class ServiceClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

### 3.3.3 Actions - Goal-Based Communication

Actions provide asynchronous, goal-based communication with feedback. This pattern is ideal for long-running operations that need to report progress and can be canceled. Examples include navigation goals, trajectory execution, or complex manipulation tasks.

```python
# Action server example
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Result: {result.sequence}')
        return result

# Action client example
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')
```

## 3.4 Quality of Service (QoS) Policies

Quality of Service (QoS) policies allow fine-tuning of communication behavior to match the requirements of specific applications. These policies are crucial for real-time systems and safety-critical applications.

### 3.4.1 Reliability Policy

The reliability policy determines whether messages are delivered reliably or best-effort:

- **Reliable**: All messages are delivered, with retries if necessary (e.g., for critical commands)
- **Best Effort**: Messages are delivered without guarantees (e.g., for sensor data where some loss is acceptable)

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy

# Reliable communication for critical commands
reliable_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE
)

# Best effort for sensor data
best_effort_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT
)
```

### 3.4.2 Durability Policy

The durability policy determines how messages are handled for late-joining subscribers:

- **Volatile**: Messages are not stored for late joiners (default, efficient)
- **Transient Local**: Messages are stored for late joiners (useful for configuration data)

```python
from rclpy.qos import QoSProfile, DurabilityPolicy

# For configuration parameters that late joiners should receive
transient_qos = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)

# For streaming data where late joiners don't need old data
volatile_qos = QoSProfile(
    depth=10,
    durability=DurabilityPolicy.VOLATILE
)
```

### 3.4.3 History Policy

The history policy determines how many messages are kept in the publisher's history:

- **Keep Last**: Keep the specified number of most recent messages
- **Keep All**: Keep all messages (use with caution due to memory usage)

```python
from rclpy.qos import QoSProfile, HistoryPolicy

# Keep last N messages
keep_last_qos = QoSProfile(
    depth=5,  # Keep last 5 messages
    history=HistoryPolicy.KEEP_LAST
)

# Keep all messages (use carefully)
keep_all_qos = QoSProfile(
    depth=0,  # Special value for keep all
    history=HistoryPolicy.KEEP_ALL
)
```

## 3.5 ROS 2 Tools for Communication Analysis

ROS 2 provides powerful tools for debugging and analyzing communication patterns in robotic systems.

### 3.5.1 Command Line Tools

- `ros2 node list`: List all active nodes
- `ros2 topic list`: List all topics
- `ros2 service list`: List all services
- `ros2 action list`: List all actions
- `ros2 topic echo <topic_name>`: Display messages on a topic
- `ros2 topic info <topic_name>`: Show information about a topic
- `ros2 node info <node_name>`: Show information about a node

### 3.5.2 Advanced Analysis Tools

- **rqt_graph**: Visualize the node graph and communication patterns
- **rqt_plot**: Plot numerical data from topics in real-time
- **rosbag2**: Record and replay ROS 2 messages for analysis
- **ros2 doctor**: Check the health of the ROS 2 system

## 3.6 Best Practices for ROS 2 Communication

### 3.6.1 Design Patterns

1. **Publisher-Subscriber for Streaming Data**: Use topics for continuous data streams like sensor readings, robot states, or camera images.

2. **Service for Request-Reply Operations**: Use services for operations that have a clear request-response pattern, like parameter setting or simple computations.

3. **Action for Long-Running Tasks**: Use actions for operations that take time to complete and may need to report progress or be canceled.

4. **Appropriate QoS for Application Requirements**: Choose QoS policies based on the real-time and reliability requirements of your application.

### 3.6.2 Performance Considerations

- **Message Size**: Large messages can impact performance; consider compression or downsampling for high-bandwidth data.
- **Frequency**: Balance update frequency with computational requirements and network capacity.
- **QoS Matching**: Ensure publisher and subscriber QoS policies are compatible to avoid communication issues.
- **Resource Management**: Monitor CPU and memory usage, especially in embedded systems.

### 3.6.3 Safety and Reliability

- **Timeout Handling**: Implement timeouts for service calls and action goals to prevent indefinite blocking.
- **Error Recovery**: Design nodes to handle communication failures gracefully.
- **Monitoring**: Implement health checks and monitoring for critical communication paths.
- **Graceful Degradation**: Design systems to continue operating in a reduced capacity when communication issues occur.

## 3.7 Integration with Physical AI Systems

ROS 2's architecture is particularly well-suited for Physical AI systems where real-time perception, reasoning, and action must be coordinated effectively.

### 3.7.1 Perception Pipeline Integration

In Physical AI systems, perception pipelines often require different communication patterns:

- **High-frequency sensor data**: Best-effort, keep-last QoS for camera and LiDAR data
- **Processed perception results**: Reliable communication for object detection and tracking
- **Calibration data**: Transient-local QoS for parameters that late-joining nodes need

### 3.7.2 Control System Integration

Control systems in Physical AI require deterministic communication:

- **Real-time control commands**: Reliable, keep-last QoS with appropriate timeouts
- **State feedback**: Reliable communication for closed-loop control
- **Safety systems**: High-priority QoS for emergency stop and safety-related messages

## Summary

ROS 2's DDS-based architecture provides a robust foundation for developing distributed robotic applications. The decentralized design eliminates single points of failure, while Quality of Service policies allow fine-tuning communication behavior for specific application requirements. Understanding the different communication patterns (topics, services, and actions) and when to use each is crucial for designing effective robotic systems. The rich set of tools available in ROS 2 enables developers to debug, analyze, and optimize their communication patterns effectively.