# Chapter 3 Quiz: ROS 2 Architecture and Communication

## Multiple Choice Questions

1. What is the primary communication middleware used in ROS 2?
   a) TCPROS/UDPROS
   b) Data Distribution Service (DDS)
   c) ZeroMQ
   d) Apache Kafka

2. Which QoS policy determines whether messages are delivered reliably or with best effort?
   a) Durability Policy
   b) History Policy
   c) Reliability Policy
   d) Deadline Policy

3. What communication pattern is most appropriate for long-running operations that need to report progress?
   a) Topics (Publish/Subscribe)
   b) Services (Request/Reply)
   c) Actions (Goal/Feedback/Result)
   d) Parameters

4. In ROS 2, what happens to a node if the master goes down (as was the case in ROS 1)?
   a) The node stops working immediately
   b) The node continues to work normally
   c) The node enters a safe mode
   d) The node automatically restarts

5. Which QoS durability policy should be used for configuration parameters that late-joining nodes need to receive?
   a) VOLATILE
   b) TRANSIENT_LOCAL
   c) KEEP_ALL
   d) BEST_EFFORT

## Practical Application Questions

6. You are designing a ROS 2 system for a mobile robot with a camera, LiDAR, and motor controllers. For each of the following communication needs, select the most appropriate communication pattern and justify your choice:
   a) Streaming camera images from the robot to a remote monitoring station
   b) Requesting the robot to navigate to a specific location
   c) Sending velocity commands to the motor controllers with real-time requirements
   d) Updating the robot's maximum speed parameter

7. Design appropriate QoS policies for the following scenarios in a ROS 2 system:
   a) Camera image streaming where some frame loss is acceptable but low latency is critical
   b) Battery status reporting where all messages must be delivered reliably
   c) Robot configuration parameters that new nodes should receive upon joining the system

## Code Analysis Questions

8. Analyze the following ROS 2 publisher code and identify potential issues:
   ```python
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image

   class CameraPublisher(Node):
       def __init__(self):
           super().__init__('camera_publisher')
           self.publisher = self.create_publisher(Image, '/camera/image_raw', 1)
           self.timer = self.create_timer(0.033, self.publish_image)  # ~30 FPS

       def publish_image(self):
           msg = Image()
           # Image data populated here
           self.publisher.publish(msg)
   ```
   What QoS considerations should be made for this publisher, and what potential issues might arise?

9. The following service client code has a potential blocking issue. Identify the problem and suggest a solution:
   ```python
   def call_service_sync(self, request):
       while not self.cli.wait_for_service(timeout_sec=1.0):
           self.get_logger().info('Service not available...')

       # This call blocks the entire node
       future = self.cli.call(request)
       return future.result()
   ```

## Conceptual Questions

10. Explain the key differences between ROS 1 and ROS 2 architectures, and describe how these differences address the challenges of deploying robotics applications in production environments.

11. Describe the role of DDS in ROS 2 and explain how it enables decentralized communication between nodes.

12. Compare and contrast the three main communication patterns in ROS 2 (topics, services, and actions). Provide specific examples of when each pattern would be most appropriate in a robotic system.

---

## Answer Key

### Multiple Choice Answers:
1. b) Data Distribution Service (DDS)
2. c) Reliability Policy
3. c) Actions (Goal/Feedback/Result)
4. b) The node continues to work normally
5. b) TRANSIENT_LOCAL

### Practical Application Answers:

6. Communication pattern selections:
   a) Topics (Publish/Subscribe) - Appropriate for continuous streaming of camera data
   b) Actions (Goal/Feedback/Result) - Navigation is a long-running task that needs to report progress and can be canceled
   c) Topics (Publish/Subscribe) - Real-time velocity commands need low-latency publishing
   d) Services (Request/Reply) - Parameter updates are discrete operations with request/response pattern

7. QoS policy designs:
   a) Best effort reliability, keep-last history with small depth, volatile durability
   b) Reliable reliability, keep-last history, volatile durability
   c) Reliable reliability, keep-last history, transient_local durability

### Code Analysis Answers:

8. Potential issues:
   - Queue size of 1 may cause message drops at 30 FPS
   - No QoS policy specified, using defaults which may not be appropriate for camera data
   - Should consider best-effort reliability and appropriate history policy for camera streaming
   - Solution: Increase queue size and specify appropriate QoS for camera streaming

9. Problem: `self.cli.call(request)` is a blocking synchronous call that will block the entire node. Solution: Use asynchronous calls with futures:
   ```python
   def call_service_async(self, request):
       while not self.cli.wait_for_service(timeout_sec=1.0):
           self.get_logger().info('Service not available...')

       future = self.cli.call_async(request)
       # Use callbacks or spin utilities instead of blocking
       return future
   ```

### Conceptual Answers:

10. Key differences between ROS 1 and ROS 2:
   - ROS 1 used a centralized master-based architecture, while ROS 2 uses a decentralized DDS-based architecture
   - ROS 2 eliminates the single point of failure by removing the master
   - ROS 2 provides real-time support through DDS QoS policies
   - ROS 2 includes built-in security features
   - ROS 2 supports multi-robot systems natively
   - ROS 2 is more suitable for deployment in production environments

11. DDS (Data Distribution Service) is the underlying communication middleware in ROS 2. It enables decentralized communication by:
   - Providing peer-to-peer communication without requiring a central broker
   - Implementing a publish-subscribe model where participants discover each other dynamically
   - Offering Quality of Service (QoS) policies for fine-tuning communication behavior
   - Supporting real-time communication with deterministic guarantees
   - Enabling fault tolerance through decentralized architecture

12. Comparison of communication patterns:
   - Topics (Publish/Subscribe): Asynchronous, one-way communication ideal for streaming data like sensor readings. No response expected.
   - Services (Request/Reply): Synchronous, bidirectional communication for discrete operations with clear request/response pattern. Blocks until response received.
   - Actions (Goal/Feedback/Result): Asynchronous, goal-based communication with progress feedback and cancellation capability. Ideal for long-running operations.