# Chapter 7: Perception Systems and Computer Vision

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the fundamental concepts of robot perception and computer vision
- Implement various computer vision algorithms for robotics applications
- Integrate perception systems with ROS 2 for real-time processing
- Apply machine learning and deep learning techniques for object detection and recognition
- Design perception pipelines for different robotic tasks and environments
- Evaluate perception system performance and handle uncertainty in sensor data

## 7.1 Introduction to Robot Perception

Robot perception is the process by which robots acquire, interpret, and understand information about their environment. This capability is fundamental to autonomous robotics, enabling robots to navigate, manipulate objects, interact with humans, and perform complex tasks in unstructured environments.

### 7.1.1 The Perception-Action Loop

In Physical AI systems, perception is tightly coupled with action through a continuous loop:

1. **Sensing**: Acquiring data from various sensors (cameras, LiDAR, IMU, etc.)
2. **Processing**: Interpreting sensor data to extract meaningful information
3. **Understanding**: Building a representation of the environment and objects
4. **Decision Making**: Determining appropriate actions based on perception
5. **Acting**: Executing actions that may change the environment
6. **Feedback**: Using the results of actions to refine perception

### 7.1.2 Types of Perception in Robotics

Robotics perception encompasses several modalities:

- **Visual Perception**: Processing camera images for object detection, recognition, and scene understanding
- **Range Perception**: Using LiDAR, depth sensors, or stereo vision for 3D scene reconstruction
- **Tactile Perception**: Sensing contact, force, and texture through touch
- **Auditory Perception**: Processing sound for localization and recognition
- **Multimodal Perception**: Combining multiple sensory modalities for robust understanding

## 7.2 Camera Systems and Image Acquisition

### 7.2.1 Camera Models and Calibration

Understanding camera models is crucial for accurate perception. The pinhole camera model is the foundation for most computer vision applications:

```
u = fx * (X/Z) + cx
v = fy * (Y/Z) + cy
```

Where (u,v) are pixel coordinates, (X,Y,Z) are 3D world coordinates, and (fx,fy) are focal lengths, (cx,cy) are principal points.

### 7.2.2 Camera Calibration Process

```python
import cv2
import numpy as np

def calibrate_camera(images, pattern_size=(9, 6)):
    """Calibrate camera using chessboard pattern"""
    # Prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Perform calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    return mtx, dist  # Camera matrix and distortion coefficients
```

### 7.2.3 Image Preprocessing

Before processing, images often require preprocessing:

```python
def preprocess_image(image, camera_matrix, dist_coeffs):
    """Preprocess image for computer vision tasks"""
    # Undistort image
    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, newcameramtx)

    # Apply ROI to remove black regions
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]

    # Convert to grayscale if needed
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

    return undistorted, gray
```

## 7.3 Feature Detection and Matching

### 7.3.1 Key Point Detection

Feature detection is fundamental for many computer vision tasks:

```python
def detect_features(image):
    """Detect and describe features in an image"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # SIFT detector (requires opencv-contrib-python)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Alternative: ORB detector (free alternative)
    # orb = cv2.ORB_create()
    # keypoints, descriptors = orb.detectAndCompute(gray, None)

    return keypoints, descriptors

def match_features(desc1, desc2):
    """Match features between two images"""
    # Create BFMatcher object
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches
```

### 7.3.2 Image Alignment and Homography

```python
def find_homography(img1, img2):
    """Find homography transformation between two images"""
    kp1, desc1 = detect_features(img1)
    kp2, desc2 = detect_features(img2)

    matches = match_features(desc1, desc2)

    if len(matches) >= 4:
        # Get matched keypoint coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography matrix
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return H, matches, mask
    else:
        return None, [], None
```

## 7.4 Object Detection and Recognition

### 7.4.1 Traditional Object Detection

Traditional methods use hand-crafted features:

```python
def detect_objects_traditional(image):
    """Detect objects using traditional computer vision methods"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    min_area = 100
    objects = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            objects.append({'bbox': (x, y, w, h), 'area': area})

    return objects

def template_matching(image, template):
    """Find template in image using template matching"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Find locations where matching exceeds threshold
    threshold = 0.8
    locations = np.where(result >= threshold)

    matches = []
    for pt in zip(*locations[::-1]):
        matches.append({'bbox': (pt[0], pt[1], template_gray.shape[1], template_gray.shape[0]),
                        'confidence': result[pt[1], pt[0]]})

    return matches
```

### 7.4.2 Deep Learning-Based Object Detection

Modern approaches use deep learning for superior performance:

```python
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class DeepObjectDetector:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold

        # Load pre-trained model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

        # Define image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def detect_objects(self, image):
        """Detect objects using deep learning model"""
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Extract results
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        # Filter by confidence
        detections = []
        for i in range(len(boxes)):
            if scores[i] > self.confidence_threshold:
                detections.append({
                    'bbox': boxes[i].astype(int),
                    'label': int(labels[i]),
                    'confidence': float(scores[i])
                })

        return detections
```

## 7.5 ROS 2 Integration for Perception

### 7.5.1 Camera Data Processing Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
import cv2
import numpy as np

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Object detector
        self.detector = DeepObjectDetector()

        # Setup subscribers and publishers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.detection_pub = self.create_publisher(
            String,  # In practice, use a custom message type
            '/object_detections',
            10
        )

        # Setup processing timer
        self.process_timer = self.create_timer(0.1, self.process_loop)

        self.current_image = None
        self.detections = []

        self.get_logger().info("Perception node initialized")

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def process_loop(self):
        """Main processing loop"""
        if self.current_image is not None:
            # Perform object detection
            detections = self.detector.detect_objects(self.current_image)
            self.detections = detections

            # Process and publish results
            if detections:
                detection_info = f"Detected {len(detections)} objects"
                detection_msg = String()
                detection_msg.data = detection_info
                self.detection_pub.publish(detection_msg)

                # Visualize detections
                self.visualize_detections()

    def visualize_detections(self):
        """Visualize detections on image"""
        if self.current_image is not None and self.detections:
            vis_image = self.current_image.copy()

            for detection in self.detections:
                bbox = detection['bbox']
                confidence = detection['confidence']

                # Draw bounding box
                cv2.rectangle(vis_image,
                             (int(bbox[0]), int(bbox[1])),
                             (int(bbox[2]), int(bbox[3])),
                             (0, 255, 0), 2)

                # Add confidence text
                cv2.putText(vis_image, f'{confidence:.2f}',
                           (int(bbox[0]), int(bbox[1])-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # For visualization, you might publish the processed image
            # processed_msg = self.cv_bridge.cv2_to_imgmsg(vis_image, "bgr8")
            # self.processed_image_pub.publish(processed_msg)
```

### 7.5.2 Multi-Sensor Fusion

```python
class MultiSensorFusion(Node):
    def __init__(self):
        super().__init__('multi_sensor_fusion')

        # Subscribers for different sensors
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10
        )

        self.lidar_sub = self.create_subscription(
            PointCloud2, '/lidar/points', self.lidar_callback, 10
        )

        # Publisher for fused perception
        self.fused_perception_pub = self.create_publisher(
            PerceptionResult, '/fused_perception', 10
        )

        # Data storage
        self.camera_data = None
        self.lidar_data = None
        self.camera_timestamp = None
        self.lidar_timestamp = None

    def camera_callback(self, msg):
        """Process camera data"""
        self.camera_data = msg
        self.camera_timestamp = msg.header.stamp

    def lidar_callback(self, msg):
        """Process LiDAR data"""
        self.lidar_data = msg
        self.lidar_timestamp = msg.header.stamp

    def fuse_sensor_data(self):
        """Fuse camera and LiDAR data"""
        if (self.camera_data is not None and
            self.lidar_data is not None and
            abs((self.camera_timestamp.sec + self.camera_timestamp.nanosec/1e9) -
                (self.lidar_timestamp.sec + self.lidar_timestamp.nanosec/1e9)) < 0.1):

            # Convert LiDAR points to camera frame
            camera_image = self.cv_bridge.imgmsg_to_cv2(self.camera_data, "bgr8")
            lidar_points = self.pointcloud2_to_array(self.lidar_data)

            # Project 3D points to 2D image
            projected_points = self.project_lidar_to_camera(lidar_points)

            # Perform fusion (simplified example)
            fused_result = self.perform_fusion(camera_image, projected_points)

            # Publish result
            self.publish_fused_result(fused_result)

    def project_lidar_to_camera(self, lidar_points):
        """Project LiDAR points to camera image coordinates"""
        # This requires camera intrinsic and extrinsic parameters
        # Simplified projection matrix
        projection_matrix = np.array([
            [1000, 0, 320],
            [0, 1000, 240],
            [0, 0, 1]
        ])

        # Project 3D points to 2D
        projected = []
        for point in lidar_points:
            if point[2] > 0:  # Only points in front of camera
                projected_point = projection_matrix @ np.array([point[0], point[1], point[2], 1])
                projected_point = projected_point[:2] / projected_point[2]  # Normalize
                projected.append(projected_point)

        return projected
```

## 7.6 3D Perception and Reconstruction

### 7.6.1 Stereo Vision

Stereo vision enables depth estimation from two cameras:

```python
class StereoProcessor:
    def __init__(self, camera_config):
        self.camera_config = camera_config

        # Create stereo matcher
        self.stereo = cv2.StereoSGBM_create(
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

    def compute_disparity(self, left_image, right_image):
        """Compute disparity map from stereo images"""
        gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        # Compute disparity
        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

        return disparity

    def reconstruct_3d(self, disparity, Q_matrix):
        """Reconstruct 3D points from disparity"""
        # Q is the reprojection matrix from stereo rectification
        points_3d = cv2.reprojectImageTo3D(disparity, Q_matrix)

        return points_3d
```

### 7.6.2 Structure from Motion (SfM)

Structure from Motion reconstructs 3D scenes from multiple 2D images:

```python
def structure_from_motion(images):
    """Perform Structure from Motion reconstruction"""
    # Extract features from all images
    all_keypoints = []
    all_descriptors = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = detect_features(img)
        all_keypoints.append(kp)
        all_descriptors.append(desc)

    # Match features between consecutive images
    matches_sequence = []
    for i in range(len(all_descriptors) - 1):
        matches = match_features(all_descriptors[i], all_descriptors[i+1])
        matches_sequence.append(matches)

    # Estimate camera poses and 3D structure
    # This is a simplified version - full SfM is more complex
    camera_poses = []
    points_3d = []

    # Initialize with first camera at origin
    camera_poses.append(np.eye(4))  # Identity matrix

    for i, matches in enumerate(matches_sequence):
        if len(matches) >= 8:  # Need at least 8 points for pose estimation
            # Get matched points
            src_pts = np.float32([all_keypoints[i][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([all_keypoints[i+1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate essential matrix
            E, mask = cv2.findEssentialMat(src_pts, dst_pts)

            # Recover pose
            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts)

                # Create transformation matrix
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = t.flatten()

                camera_poses.append(pose)

    return camera_poses, points_3d
```

## 7.7 Deep Learning for Perception

### 7.7.1 Semantic Segmentation

Semantic segmentation provides pixel-level object classification:

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class SemanticSegmentation(nn.Module):
    def __init__(self, num_classes=21):  # Pascal VOC has 21 classes
        super(SemanticSegmentation, self).__init__()

        # Using a simplified U-Net-like architecture
        self.encoder = torch.hub.load('pytorch/vision:v0.10.0',
                                     'resnet50',
                                     pretrained=True)

        # Modify the final layer for segmentation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Extract features using encoder
        features = self.encoder(x)

        # Decode features to segmentation map
        segmentation = self.decoder(features)

        return segmentation

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')

        self.cv_bridge = CvBridge()

        # Load pre-trained segmentation model
        self.model = SemanticSegmentation()
        self.model.eval()

        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        self.segmentation_pub = self.create_publisher(Image, '/segmentation', 10)

        # Define color map for visualization
        self.color_map = self.create_color_map()

    def image_callback(self, msg):
        """Process image for semantic segmentation"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

            # Preprocess image
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((480, 640)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            input_tensor = transform(cv_image).unsqueeze(0)

            # Perform segmentation
            with torch.no_grad():
                output = self.model(input_tensor)
                predicted = torch.argmax(output, dim=1)

            # Convert to color image for visualization
            segmented_image = self.apply_color_map(predicted.squeeze().cpu().numpy())

            # Publish result
            result_msg = self.cv_bridge.cv2_to_imgmsg(segmented_image, "bgr8")
            self.segmentation_pub.publish(result_msg)

        except Exception as e:
            self.get_logger().error(f'Segmentation error: {e}')

    def create_color_map(self):
        """Create color map for different classes"""
        # Pascal VOC color map
        color_map = np.array([
            [0, 0, 0],         # background
            [128, 0, 0],       # aeroplane
            [0, 128, 0],       # bicycle
            [128, 128, 0],     # bird
            [0, 0, 128],       # boat
            [128, 0, 128],     # bottle
            [0, 128, 128],     # bus
            [128, 128, 128],   # car
            [64, 0, 0],        # cat
            [192, 0, 0],       # chair
            [64, 128, 0],      # cow
            [192, 128, 0],     # diningtable
            [64, 0, 128],      # dog
            [192, 0, 128],     # horse
            [64, 128, 128],    # motorbike
            [192, 128, 128],   # person
            [0, 64, 0],        # potted plant
            [128, 64, 0],      # sheep
            [0, 192, 0],       # sofa
            [128, 192, 0],     # train
            [0, 64, 128]       # tv/monitor
        ], dtype=np.uint8)

        return color_map

    def apply_color_map(self, segmentation_mask):
        """Apply color map to segmentation mask"""
        height, width = segmentation_mask.shape
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

        for class_idx in np.unique(segmentation_mask):
            mask = segmentation_mask == class_idx
            colored_mask[mask] = self.color_map[class_idx]

        return colored_mask
```

## 7.8 Performance Evaluation and Uncertainty

### 7.8.1 Perception Quality Metrics

Evaluating perception system performance is crucial:

```python
def evaluate_detection_performance(ground_truth, predictions, iou_threshold=0.5):
    """Evaluate object detection performance"""
    # Calculate IoU for each prediction-ground truth pair
    ious = calculate_ious(ground_truth, predictions)

    # Determine true positives, false positives, false negatives
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives

    matched_gt = set()

    # For each prediction, find best matching ground truth
    for pred_idx, pred in enumerate(predictions):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truth):
            if ious[pred_idx][gt_idx] > best_iou:
                best_iou = ious[pred_idx][gt_idx]
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def calculate_ious(boxes1, boxes2):
    """Calculate IoU matrix between two sets of bounding boxes"""
    ious = np.zeros((len(boxes1), len(boxes2)))

    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            iou = calculate_iou(box1, box2)
            ious[i, j] = iou

    return ious

def calculate_iou(box1, box2):
    """Calculate Intersection over Union for two bounding boxes"""
    # Box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0
```

### 7.8.2 Uncertainty Quantification

Quantifying uncertainty in perception results:

```python
class UncertaintyEstimator:
    def __init__(self):
        self.uncertainty_models = {}

    def estimate_detection_uncertainty(self, detection_result, input_image):
        """Estimate uncertainty for object detection results"""
        # Method 1: Monte Carlo Dropout
        uncertainty_scores = self.monte_carlo_dropout_estimate(detection_result, input_image)

        # Method 2: Ensemble prediction
        ensemble_uncertainty = self.ensemble_estimate(detection_result, input_image)

        # Combine uncertainties
        final_uncertainty = 0.5 * uncertainty_scores + 0.5 * ensemble_uncertainty

        return final_uncertainty

    def monte_carlo_dropout_estimate(self, detection_result, input_image):
        """Estimate uncertainty using Monte Carlo dropout"""
        # This would involve running the model multiple times
        # with dropout enabled during inference
        pass

    def ensemble_estimate(self, detection_result, input_image):
        """Estimate uncertainty using model ensemble"""
        # This would involve running multiple models and measuring disagreement
        pass
```

## 7.9 Integration with Physical AI Systems

Perception systems in Physical AI must handle real-world challenges:

### 7.9.1 Robust Perception in Dynamic Environments

```python
class RobustPerceptionNode(Node):
    def __init__(self):
        super().__init__('robust_perception_node')

        # Adaptive parameters based on environment
        self.lighting_condition = 'unknown'
        self.camera_motion = 'static'

        # Setup subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.adaptive_image_callback, 10
        )

        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

    def adaptive_image_callback(self, msg):
        """Adapt perception based on environmental conditions"""
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

        # Assess lighting conditions
        lighting_score = self.assess_lighting(cv_image)

        # Adjust processing based on conditions
        if lighting_score < 0.3:  # Low light
            # Apply image enhancement
            enhanced_image = self.enhance_low_light(cv_image)
            # Use different detection parameters
            detections = self.detect_with_enhancement(enhanced_image)
        else:
            detections = self.detect_objects(cv_image)

        # If camera is moving, use motion compensation
        if self.camera_motion == 'moving':
            detections = self.compensate_motion(detections)

        # Publish results
        self.publish_detections(detections)

    def assess_lighting(self, image):
        """Assess lighting conditions in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        # Normalize to [0, 1] where 1 is good lighting
        lighting_score = min(mean_brightness / 255.0, 1.0)

        return lighting_score

    def enhance_low_light(self, image):
        """Enhance image for low-light conditions"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)

        enhanced_lab = cv2.merge([l, a, b])
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_image

    def imu_callback(self, msg):
        """Update camera motion state from IMU"""
        # Check if there's significant angular velocity
        angular_velocity = np.linalg.norm([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        if angular_velocity > 0.1:  # Threshold for motion
            self.camera_motion = 'moving'
        else:
            self.camera_motion = 'static'
```

## Summary

Robot perception and computer vision form the sensory foundation of Physical AI systems. Modern perception systems combine traditional computer vision techniques with deep learning approaches to achieve robust performance across various environments and conditions. The integration with ROS 2 enables real-time processing and multi-sensor fusion, while uncertainty quantification ensures safe operation in uncertain environments. As robots become more autonomous, perception systems must continue to evolve to handle increasingly complex and dynamic real-world scenarios.