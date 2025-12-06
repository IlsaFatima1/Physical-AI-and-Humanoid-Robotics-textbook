"""
Perception Systems and Computer Vision Examples
Chapter 7: Perception Systems and Computer Vision

This module demonstrates various perception and computer vision concepts including:
- Camera calibration and image preprocessing
- Feature detection and matching
- Object detection using traditional and deep learning methods
- ROS 2 integration for real-time perception
- Multi-sensor fusion
- 3D reconstruction and depth estimation
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from collections import deque
import threading
import time


class CameraCalibrationNode(Node):
    """
    Demonstrates camera calibration and image preprocessing techniques.
    """
    def __init__(self):
        super().__init__('camera_calibration_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Calibration parameters (these would be computed during actual calibration)
        self.camera_matrix = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0]
        ])

        self.dist_coeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.0])

        # Setup subscribers and publishers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        self.calibrated_image_pub = self.create_publisher(
            Image, '/camera/image_calibrated', 10
        )

        # Setup processing timer
        self.process_timer = self.create_timer(0.1, self.process_loop)

        self.current_image = None
        self.calibrated_image = None

        self.get_logger().info("Camera Calibration Node initialized")

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def process_loop(self):
        """Main processing loop for calibration"""
        if self.current_image is not None:
            # Apply camera calibration (undistortion)
            h, w = self.current_image.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
            )

            undistorted = cv2.undistort(
                self.current_image,
                self.camera_matrix,
                self.dist_coeffs,
                None,
                new_camera_matrix
            )

            # Apply ROI to remove black regions
            x, y, w, h = roi
            self.calibrated_image = undistorted[y:y+h, x:x+w]

            # Publish calibrated image
            calibrated_msg = self.cv_bridge.cv2_to_imgmsg(self.calibrated_image, "bgr8")
            calibrated_msg.header = self.current_image.header
            self.calibrated_image_pub.publish(calibrated_msg)


class FeatureDetectionNode(Node):
    """
    Demonstrates feature detection and matching techniques.
    """
    def __init__(self):
        super().__init__('feature_detection_node')

        self.cv_bridge = CvBridge()

        # Setup subscribers and publishers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        self.features_pub = self.create_publisher(
            Image, '/features_visualization', 10
        )

        self.features_info_pub = self.create_publisher(
            String, '/features_info', 10
        )

        # Setup processing timer
        self.process_timer = self.create_timer(0.5, self.process_features)

        self.current_image = None
        self.feature_points = []

        self.get_logger().info("Feature Detection Node initialized")

    def image_callback(self, msg):
        """Process incoming image for feature detection"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def process_features(self):
        """Process features in the current image"""
        if self.current_image is not None:
            # Convert to grayscale
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

            # Detect features using ORB (free alternative to SIFT)
            orb = cv2.ORB_create(nfeatures=500)
            keypoints, descriptors = orb.detectAndCompute(gray, None)

            if keypoints is not None:
                # Draw keypoints on image
                feature_image = cv2.drawKeypoints(
                    self.current_image, keypoints, None,
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                )

                # Publish visualization
                feature_msg = self.cv_bridge.cv2_to_imgmsg(feature_image, "bgr8")
                feature_msg.header = self.current_image.header
                self.features_pub.publish(feature_msg)

                # Publish feature information
                info_msg = String()
                info_msg.data = f"Detected {len(keypoints)} features"
                self.features_info_pub.publish(info_msg)

                self.feature_points = [(kp.pt[0], kp.pt[1]) for kp in keypoints]


class ObjectDetectionNode(Node):
    """
    Demonstrates object detection using both traditional and deep learning methods.
    """
    def __init__(self):
        super().__init__('object_detection_node')

        self.cv_bridge = CvBridge()

        # Setup subscribers and publishers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        self.detection_pub = self.create_publisher(
            Image, '/detection_visualization', 10
        )

        self.detection_info_pub = self.create_publisher(
            String, '/detection_info', 10
        )

        # Setup processing timer
        self.process_timer = self.create_timer(0.1, self.process_detections)

        self.current_image = None
        self.detections = []

        self.get_logger().info("Object Detection Node initialized")

    def image_callback(self, msg):
        """Process incoming image for object detection"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def process_detections(self):
        """Process object detection in current image"""
        if self.current_image is not None:
            # Traditional method: detect shapes using contours
            traditional_detections = self.traditional_detection(self.current_image)

            # For demonstration, also simulate deep learning detection
            deep_detections = self.simulate_deep_learning_detection(self.current_image)

            # Combine detections
            self.detections = traditional_detections + deep_detections

            # Visualize detections
            vis_image = self.visualize_detections(self.current_image.copy(), self.detections)

            # Publish visualization
            vis_msg = self.cv_bridge.cv2_to_imgmsg(vis_image, "bgr8")
            vis_msg.header = self.current_image.header
            self.detection_pub.publish(vis_msg)

            # Publish detection information
            info_msg = String()
            info_msg.data = f"Detected {len(self.detections)} objects (traditional: {len(traditional_detections)}, deep: {len(deep_detections)})"
            self.detection_info_pub.publish(info_msg)

    def traditional_detection(self, image):
        """Traditional object detection using contours and shape analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        min_area = 100  # Minimum area threshold

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate aspect ratio to filter out thin shapes
                aspect_ratio = float(w) / h if h != 0 else 0

                if 0.2 < aspect_ratio < 5.0:  # Reasonable aspect ratio
                    detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': 0.7,  # Simulated confidence
                        'class': 'shape',
                        'method': 'traditional'
                    })

        return detections

    def simulate_deep_learning_detection(self, image):
        """Simulate deep learning detection (in real implementation, this would use a trained model)"""
        # This is a simulation - in real implementation, this would use a deep learning model
        # like YOLO, SSD, or Faster R-CNN

        # For simulation, let's detect some regions of interest based on color
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for red color (for simulation purposes)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # Larger area threshold for simulated "deep" detection
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': 0.9,  # Higher confidence for simulated deep detection
                    'class': 'red_object',
                    'method': 'deep_learning'
                })

        return detections

    def visualize_detections(self, image, detections):
        """Visualize detections on image"""
        vis_image = image.copy()

        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            method = detection['method']

            # Choose color based on detection method
            color = (0, 255, 0) if method == 'deep_learning' else (255, 0, 0)

            # Draw bounding box
            cv2.rectangle(vis_image,
                         (bbox[0], bbox[1]),
                         (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                         color, 2)

            # Add label
            label = f"{detection['class']} ({confidence:.2f})"
            cv2.putText(vis_image, label,
                       (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return vis_image


class StereoDepthNode(Node):
    """
    Demonstrates stereo vision and depth estimation.
    """
    def __init__(self):
        super().__init__('stereo_depth_node')

        self.cv_bridge = CvBridge()

        # Setup subscribers for stereo pair
        self.left_image_sub = self.create_subscription(
            Image, '/stereo/left/image_rect_color', self.left_image_callback, 10
        )

        self.right_image_sub = self.create_subscription(
            Image, '/stereo/right/image_rect_color', self.right_image_callback, 10
        )

        self.depth_pub = self.create_publisher(
            Image, '/stereo/depth_map', 10
        )

        self.point_cloud_pub = self.create_publisher(
            PointCloud2, '/stereo/point_cloud', 10
        )

        # Stereo matcher
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=96,
            blockSize=11,
            P1=8 * 3 * 11**2,
            P2=32 * 3 * 11**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Store stereo pair
        self.left_image = None
        self.right_image = None
        self.left_timestamp = None
        self.right_timestamp = None

        # Setup processing timer
        self.process_timer = self.create_timer(0.2, self.process_stereo)

        self.get_logger().info("Stereo Depth Node initialized")

    def left_image_callback(self, msg):
        """Process left stereo image"""
        try:
            self.left_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.left_timestamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Error processing left image: {e}')

    def right_image_callback(self, msg):
        """Process right stereo image"""
        try:
            self.right_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.right_timestamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Error processing right image: {e}')

    def process_stereo(self):
        """Process stereo pair to compute depth"""
        if (self.left_image is not None and
            self.right_image is not None and
            self.left_timestamp is not None and
            self.right_timestamp is not None):

            # Check if images are approximately synchronized
            time_diff = abs(
                (self.left_timestamp.sec + self.left_timestamp.nanosec/1e9) -
                (self.right_timestamp.sec + self.right_timestamp.nanosec/1e9)
            )

            if time_diff < 0.05:  # 50ms tolerance
                # Compute disparity map
                gray_left = cv2.cvtColor(self.left_image, cv2.COLOR_BGR2GRAY)
                gray_right = cv2.cvtColor(self.right_image, cv2.COLOR_BGR2GRAY)

                disparity = self.stereo_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0

                # Normalize disparity for visualization
                disp_normalized = cv2.normalize(disparity, None, alpha=0, beta=255,
                                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # Publish depth map
                depth_msg = self.cv_bridge.cv2_to_imgmsg(disp_normalized, "mono8")
                depth_msg.header = self.left_image.header
                self.depth_pub.publish(depth_msg)

                # Convert to point cloud (simplified)
                # In real implementation, you would use the Q matrix from stereo calibration
                self.publish_point_cloud(disparity, self.left_image)

    def publish_point_cloud(self, disparity, left_image):
        """Convert disparity to point cloud (simplified)"""
        # This is a simplified version - real implementation would use proper Q matrix
        height, width = disparity.shape

        # Generate simple point cloud
        points = []
        colors = []

        # Sample points for efficiency
        step = 10  # Only process every 10th pixel

        for y in range(0, height, step):
            for x in range(0, width, step):
                if 0 <= x < width and 0 <= y < height:
                    depth_val = disparity[y, x]

                    # Only include points with reasonable depth
                    if 0 < depth_val < 200:  # Filter out invalid disparities
                        # Convert to 3D point (simplified)
                        # In real implementation, use Q matrix from stereo calibration
                        z = 1000.0 / (depth_val + 1e-6)  # Simplified depth calculation
                        x_3d = (x - width/2) * z / 500.0  # Approximate using focal length
                        y_3d = (y - height/2) * z / 500.0

                        points.append([x_3d, y_3d, z])

                        # Get color from image
                        if len(left_image.shape) == 3:
                            b, g, r = left_image[y, x]
                            colors.append([r, g, b])

        # In a real implementation, you would create a proper PointCloud2 message
        # For now, we'll just log the number of points
        self.get_logger().info(f"Generated point cloud with {len(points)} points")


class MultiSensorFusionNode(Node):
    """
    Demonstrates fusion of multiple sensor modalities for robust perception.
    """
    def __init__(self):
        super().__init__('multi_sensor_fusion_node')

        self.cv_bridge = CvBridge()

        # Setup subscribers for different sensors
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10
        )

        self.lidar_sub = self.create_subscription(
            PointCloud2, '/lidar/points', self.lidar_callback, 10
        )

        self.imu_sub = self.create_subscription(
            Float32MultiArray, '/imu/data', self.imu_callback, 10
        )

        # Publishers for fused results
        self.fused_detection_pub = self.create_publisher(
            String, '/fused_detections', 10
        )

        self.fused_visualization_pub = self.create_publisher(
            Image, '/fused_visualization', 10
        )

        # Data buffers
        self.camera_buffer = deque(maxlen=5)
        self.lidar_buffer = deque(maxlen=5)
        self.imu_buffer = deque(maxlen=10)

        # Setup processing timer
        self.process_timer = self.create_timer(0.1, self.fuse_sensors)

        self.get_logger().info("Multi-Sensor Fusion Node initialized")

    def camera_callback(self, msg):
        """Process camera data"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.camera_buffer.append({
                'image': cv_image,
                'timestamp': msg.header.stamp,
                'header': msg.header
            })
        except Exception as e:
            self.get_logger().error(f'Error processing camera data: {e}')

    def lidar_callback(self, msg):
        """Process LiDAR data"""
        # In a real implementation, you would convert PointCloud2 to numpy array
        # For simulation, we'll just store the message
        self.lidar_buffer.append({
            'data': msg,
            'timestamp': msg.header.stamp
        })

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_buffer.append({
            'data': msg.data,
            'timestamp': self.get_clock().now().to_msg()
        })

    def fuse_sensors(self):
        """Fuse data from multiple sensors"""
        if (len(self.camera_buffer) > 0 and
            len(self.lidar_buffer) > 0 and
            len(self.imu_buffer) > 0):

            # Get latest data
            latest_camera = self.camera_buffer[-1]
            latest_lidar = self.lidar_buffer[-1]
            latest_imu = self.imu_buffer[-1]

            # Calculate time differences
            camera_time = latest_camera['timestamp']
            lidar_time = latest_lidar['timestamp']

            time_diff = abs(
                (camera_time.sec + camera_time.nanosec/1e9) -
                (lidar_time.sec + lidar_time.nanosec/1e9)
            )

            # Only fuse if data is synchronized (within 100ms)
            if time_diff < 0.1:
                # Perform fusion - this is a simplified example
                fused_result = self.perform_fusion(
                    latest_camera['image'],
                    latest_lidar['data'],
                    latest_imu['data']
                )

                # Publish fused result
                result_msg = String()
                result_msg.data = f"Fused perception result at time diff: {time_diff:.3f}s"
                self.fused_detection_pub.publish(result_msg)

                # Visualize fusion result
                vis_image = self.visualize_fusion(
                    latest_camera['image'],
                    fused_result
                )

                vis_msg = self.cv_bridge.cv2_to_imgmsg(vis_image, "bgr8")
                vis_msg.header = latest_camera['header']
                self.fused_visualization_pub.publish(vis_msg)

    def perform_fusion(self, camera_image, lidar_data, imu_data):
        """Perform actual sensor fusion (simplified)"""
        # In a real implementation, this would:
        # 1. Project LiDAR points to camera image coordinates
        # 2. Combine detections from both sensors
        # 3. Use IMU data for motion compensation
        # 4. Apply probabilistic fusion methods

        # For simulation, return a simple fusion indicator
        return {
            'camera_detections': self.detect_from_camera(camera_image),
            'lidar_detections': self.detect_from_lidar(lidar_data),
            'confidence': 0.9
        }

    def detect_from_camera(self, image):
        """Detect objects from camera image"""
        # Use the same detection method as ObjectDetectionNode
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append((x, y, w, h))

        return detections

    def detect_from_lidar(self, lidar_data):
        """Detect objects from LiDAR data (simulated)"""
        # In real implementation, process PointCloud2 data
        # For simulation, return fixed number of detections
        return 5  # Simulated LiDAR detections

    def visualize_fusion(self, image, fusion_result):
        """Visualize fusion result"""
        vis_image = image.copy()

        # Draw camera detections
        for detection in fusion_result['camera_detections']:
            x, y, w, h = detection
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Add fusion indicator
        cv2.putText(vis_image, f"Fusion Confidence: {fusion_result['confidence']:.2f}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis_image


def main(args=None):
    """Main function to run perception system nodes"""
    rclpy.init(args=args)

    # Create all perception nodes
    calibration_node = CameraCalibrationNode()
    feature_node = FeatureDetectionNode()
    detection_node = ObjectDetectionNode()
    stereo_node = StereoDepthNode()
    fusion_node = MultiSensorFusionNode()

    try:
        # Create executor and add all nodes
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(calibration_node)
        executor.add_node(feature_node)
        executor.add_node(detection_node)
        executor.add_node(stereo_node)
        executor.add_node(fusion_node)

        # Spin all nodes
        executor.spin()
    except KeyboardInterrupt:
        calibration_node.get_logger().info('Calibration node interrupted')
        feature_node.get_logger().info('Feature node interrupted')
        detection_node.get_logger().info('Detection node interrupted')
        stereo_node.get_logger().info('Stereo node interrupted')
        fusion_node.get_logger().info('Fusion node interrupted')
    finally:
        calibration_node.destroy_node()
        feature_node.destroy_node()
        detection_node.destroy_node()
        stereo_node.destroy_node()
        fusion_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()