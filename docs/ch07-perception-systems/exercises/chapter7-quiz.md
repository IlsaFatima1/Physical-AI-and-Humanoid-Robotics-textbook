# Chapter 7 Quiz: Perception Systems and Computer Vision

## Multiple Choice Questions

1. What is the primary purpose of camera calibration in robotics perception?
   a) To improve image resolution
   b) To correct lens distortion and establish the relationship between pixel and world coordinates
   c) To increase frame rate
   d) To reduce image noise

2. Which of the following is NOT a typical component of the perception-action loop?
   a) Sensing
   b) Processing
   c) Memorizing
   d) Acting

3. What does IoU (Intersection over Union) measure in object detection?
   a) Image brightness
   b) The overlap between predicted and ground truth bounding boxes
   c) Camera focal length
   d) Processing speed

4. Which deep learning architecture is commonly used for semantic segmentation?
   a) ResNet
   b) U-Net
   c) LSTM
   d) GRU

5. What is the main advantage of stereo vision over monocular vision?
   a) Higher resolution
   b) Lower computational cost
   c) Direct depth estimation
   d) Better color representation

## Practical Application Questions

6. You are developing a perception system for a mobile robot that needs to detect and avoid obstacles in real-time. Design a complete pipeline that includes:
   a) Preprocessing steps to handle varying lighting conditions
   b) Object detection approach (traditional vs. deep learning)
   c) How to integrate the perception output with navigation planning
   d) Methods to handle uncertainty in detections

7. Implement a multi-sensor fusion approach that combines camera and LiDAR data for robust object detection. Describe:
   a) How to align the coordinate systems of different sensors
   b) The fusion algorithm you would use
   c) How to handle temporal synchronization issues
   d) Validation methods for the fused output

8. Design a semantic segmentation system for an autonomous robot that operates in indoor environments. Consider:
   a) Which classes to include in the segmentation
   b) Network architecture choices
   c) Training data requirements
   d) Real-time performance optimization

## Code Analysis Questions

9. Analyze the following object detection code and identify potential issues:
   ```python
   def detect_objects(image):
       gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

       # Simple thresholding approach
       _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

       # Find contours
       contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

       objects = []
       for contour in contours:
           area = cv2.contourArea(contour)
           if area > 50:  # Fixed threshold
               x, y, w, h = cv2.boundingRect(contour)
               objects.append({'bbox': (x, y, w, h)})

       return objects
   ```

10. The following ROS 2 perception node has potential performance issues. Identify and suggest improvements:
    ```python
    class PerceptionNode(Node):
        def __init__(self):
            super().__init__('perception_node')
            self.cv_bridge = CvBridge()

            self.image_sub = self.create_subscription(
                Image, '/camera/image_raw', self.image_callback, 1
            )

        def image_callback(self, msg):
            # Process image synchronously (blocking)
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

            # Heavy processing that blocks the callback
            results = self.heavy_processing(cv_image)

            # Publish results
            self.publish_results(results)

            # No rate limiting - processes every image
    ```

## Conceptual Questions

11. Explain the differences between semantic segmentation, instance segmentation, and panoptic segmentation. When would you use each approach in robotics applications?

12. Describe the challenges of performing perception in dynamic environments and propose solutions to handle moving objects, changing lighting, and camera motion.

13. How does uncertainty quantification improve the safety and reliability of robotic perception systems? What methods would you use to estimate uncertainty?

14. Discuss the trade-offs between traditional computer vision methods and deep learning approaches for robotics perception. When would you choose one over the other?

---

## Answer Key

### Multiple Choice Answers:
1. b) To correct lens distortion and establish the relationship between pixel and world coordinates
2. c) Memorizing
3. b) The overlap between predicted and ground truth bounding boxes
4. b) U-Net
5. c) Direct depth estimation

### Practical Application Answers:

6. Real-time obstacle detection pipeline:
   a) Preprocessing: Histogram equalization, adaptive thresholding, noise reduction
   b) Approach: Deep learning (YOLO or SSD) for accuracy, traditional methods as fallback
   c) Integration: Publish detections to costmap_2d for navigation stack
   d) Uncertainty: Confidence thresholds, temporal consistency checks, sensor fusion

7. Multi-sensor fusion approach:
   a) Alignment: Extrinsic calibration to find transformation matrix between sensors
   b) Algorithm: Probabilistic fusion using Kalman filters or particle filters
   c) Synchronization: Timestamp interpolation and buffer management
   d) Validation: Cross-validation between sensors, consistency checks

8. Semantic segmentation system:
   a) Classes: Walls, floor, furniture, people, robots, obstacles
   b) Architecture: Efficient encoder-decoder like MobileNet + U-Net for real-time
   c) Data: Synthetic data generation + real-world annotated images
   d) Optimization: Model quantization, pruning, and hardware acceleration

### Code Analysis Answers:

9. Issues with the object detection code:
   - Fixed threshold doesn't adapt to lighting conditions
   - No noise filtering before thresholding
   - Fixed area threshold may not work for different object sizes
   - No validation of geometric properties
   - Improvements: Adaptive thresholding, morphological operations, aspect ratio checks

10. Issues with the ROS 2 node:
   - Heavy processing in callback blocks message handling
   - No rate limiting causes excessive processing
   - Fixed queue size of 1 may drop messages
   - Improvements: Use separate processing thread, rate limiting, larger queue size:
   ```python
   def image_callback(self, msg):
       # Just store image for processing
       self.latest_image = msg

   def processing_timer_callback(self):
       if self.latest_image is not None:
           # Process in separate thread or use asyncio
           self.process_image_async(self.latest_image)
   ```

### Conceptual Answers:

11. Segmentation differences:
   - Semantic segmentation: Pixel-level classification without instance separation
   - Instance segmentation: Distinguishes between different instances of same class
   - Panoptic segmentation: Combines semantic and instance segmentation
   Use semantic for scene understanding, instance for object manipulation, panoptic for comprehensive scene analysis

12. Dynamic environment challenges:
   - Moving objects: Background subtraction, optical flow, temporal differencing
   - Changing lighting: Adaptive algorithms, image enhancement, multiple models
   - Camera motion: Motion compensation, visual-inertial fusion, motion blur reduction

13. Uncertainty quantification benefits:
   - Enables safe decision-making under uncertainty
   - Helps determine when to request human intervention
   - Methods: Monte Carlo dropout, ensemble methods, Bayesian approaches
   - Allows dynamic confidence threshold adjustment

14. Traditional vs. deep learning trade-offs:
   - Traditional: Interpretable, less data required, faster inference, limited complexity
   - Deep learning: Better accuracy on complex tasks, requires large datasets, computationally intensive
   - Choose traditional for simple, well-defined tasks; deep learning for complex perception