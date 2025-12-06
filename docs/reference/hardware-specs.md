# Hardware Specifications and Reference Tables

This reference provides detailed specifications for common robotics hardware platforms, sensors, actuators, and computing systems used in Physical AI and Humanoid Robotics.

## Computing Platforms

### NVIDIA Jetson Series

| Model | GPU | CPU | RAM | Power (W) | Use Case | ROS Compatibility |
|-------|-----|-----|-----|----------|----------|-------------------|
| Jetson Nano | 128-core Maxwell | Quad-core ARM A57 | 4GB LPDDR4 | 5-10 | Entry-level AI | ROS 2 Humble |
| Jetson TX2 | 256-core Pascal | Dual Denver 2 + Quad ARM A57 | 8GB LPDDR4 | 7-15 | Mobile robotics | ROS 2 Humble |
| Jetson Xavier NX | 384-core Volta | Hex-core ARM Carmel | 8GB LPDDR4x | 10-15 | Edge AI | ROS 2 Humble |
| Jetson AGX Xavier | 512-core Volta | Octo-core ARM Carmel | 32GB LPDDR4x | 10-30 | High-performance | ROS 2 Humble |
| Jetson AGX Orin | 2048-core Ada | 12-core ARM Hercules | 64GB LPDDR5x | 15-60 | Advanced robotics | ROS 2 Humble |

### Single Board Computers

| Platform | CPU | RAM | GPU | Power (W) | ROS Compatibility | Notes |
|----------|-----|-----|-----|----------|-------------------|-------|
| Raspberry Pi 4 | Quad-core ARM Cortex-A72 | 2GB/4GB/8GB | VideoCore VI | 3-7 | Limited (lightweight) | Good for basic tasks |
| Coral Dev Board | ARM Cortex-A53 + Edge TPU | 1GB/4GB | Mali-G71 | 5-10 | ROS 2 (limited) | AI acceleration |
| UP Board | Quad-core x86 | 2GB-8GB | Intel HD | 6-15 | Full ROS 2 support | x86 compatibility |

## Sensors

### Cameras

| Type | Model | Resolution | FPS | Interface | ROS Package | Use Case |
|------|-------|------------|-----|-----------|-------------|----------|
| RGB | Intel RealSense D435 | 1280×720 | 90 | USB 3.0 | realsense2_camera | Depth sensing |
| RGB | Intel RealSense D435i | 1280×720 | 90 | USB 3.0 | realsense2_camera | IMU + depth |
| RGB | Intel RealSense D455 | 1280×720 | 90 | USB 3.0 | realsense2_camera | High-accuracy depth |
| Stereo | ZED 2 | 2208×1242 | 60 | USB 3.0 | zed_wrapper | 3D perception |
| Thermal | FLIR Lepton 3.5 | 160×120 | 9 | SPI | flir_lepton | Thermal imaging |

### IMU Sensors

| Model | Type | Accel Range (g) | Gyro Range (dps) | Interface | ROS Package | Notes |
|-------|------|-----------------|------------------|-----------|-------------|-------|
| Bosch BNO055 | 9-DOF | ±2,4,8,16 | ±2000 | I2C/SPI | imu_bno055 | Absolute orientation |
| ST LSM6DSOX | 6-DOF | ±2,4,8,16 | ±125,250,500,1000,2000 | I2C/SPI | imu_lsm6ds | High-performance |
| ADIS16470 | 6-DOF | ±180 | ±2000 | SPI | imu_adis16470 | Tactical grade |

### LiDAR Sensors

| Model | Range (m) | FOV (H×V) | Accuracy (cm) | Rate (Hz) | Interface | ROS Package |
|-------|-----------|-----------|---------------|-----------|-----------|-------------|
| SICK TiM571 | 10 | 270°×69.9° | ±2 | 15.6 | Ethernet | sick_tim | Indoor mapping |
| Hokuyo UST-10LX | 10 | 270°×4° | ±3 | 40 | Ethernet | hokuyo_node | Reliable indoor |
| Velodyne VLP-16 | 100 | 360°×15.6° | ±3 | 5-20 | Ethernet | velodyne | Outdoor mapping |
| Slamtec RPLidar A2 | 12 | 360°×360° | ±3 | 5-10 | USB | rplidar_ros | Cost-effective |

## Actuators and Motors

### Servo Motors

| Model | Torque (kg·cm) | Speed (0.1s/60°) | Weight (g) | Interface | Use Case |
|-------|-----------------|------------------|------------|-----------|----------|
| Dynamixel AX-12A | 1.5 | 0.17 | 51.5 | Serial | Robot arms |
| Dynamixel XL-320 | 0.39 | 0.11 | 16.7 | Serial | Low-power |
| Dynamixel XM430-W350 | 3.5 | 0.22 | 55.5 | Serial | High-torque |
| Futaba S3003 | 3.9 | 0.23 | 55 | PWM | Standard servo |

### Brushless DC Motors

| Model | Power (W) | Speed (RPM) | Torque (N·m) | Controller | Use Case |
|-------|-----------|-------------|--------------|------------|----------|
| T-Motor MN5208 | 800 | 340 | 2.24 | BLHeli_32 | Propulsion |
| DJI 2212 | 200 | 990 | 0.19 | DJI ESC | Precision control |
| KDE Direct 2814XF | 1000 | 500 | 1.91 | KDE ESC | Heavy lifting |

### Linear Actuators

| Type | Force (N) | Speed (mm/s) | Stroke (mm) | Control | Use Case |
|------|-----------|---------------|-------------|---------|----------|
| Servo-driven | 100-500 | 5-20 | 50-200 | PWM | Precise positioning |
| DC-motor-driven | 200-1000 | 10-50 | 100-500 | PWM/Voltage | Heavy-duty |

## Robot Platforms

### Mobile Base Platforms

| Platform | Drive | Payload (kg) | Max Speed (m/s) | Size (L×W×H mm) | Use Case |
|----------|-------|--------------|-----------------|------------------|----------|
| TurtleBot 4 | Differential | 2 | 1.0 | 360×360×410 | Education |
| Clearpath Jackal | Ackermann | 25 | 2.0 | 600×400×300 | Research |
| Clearpath Husky | Skid-steer | 75 | 1.5 | 800×600×300 | Outdoors |
| MiR100 | Differential | 100 | 1.2 | 830×595×420 | Logistics |

### Manipulator Arms

| Model | DOF | Reach (mm) | Payload (kg) | Accuracy (mm) | ROS Support | Use Case |
|-------|-----|------------|--------------|---------------|-------------|----------|
| UR3e | 6 | 500 | 3 | ±0.03 | Universal_Robots_ROS_Driver | Light assembly |
| Kinova Gen3 | 7 | 900 | 0.5 | ±0.1 | kinova-ros | Service robotics |
| Franka Emika Panda | 7 | 850 | 3 | ±0.1 | franka_ros | Research |
| WidowX 250 | 5 | 250 | 0.5 | ±1.0 | interbotix_ros_manipulators | Education |

### Humanoid Platforms

| Platform | Height (cm) | DOF | Weight (kg) | ROS Support | Use Case |
|----------|-------------|-----|-------------|-------------|----------|
| NAO | 58 | 25 | 5.2 | naoqi_driver | Education |
| Pepper | 120 | 20 | 28 | naoqi_driver | Service |
| Romeo | 140 | 37 | 19 | ros_romeo | Research |
| HRP-4 | 158 | 46 | 45 | hrpsys_ros_bridge | Research |

## Communication Modules

### Wireless Communication

| Type | Range | Band | Data Rate | Power (W) | ROS Integration |
|------|-------|------|-----------|-----------|-----------------|
| WiFi (802.11n) | 100m indoor | 2.4/5GHz | 150 Mbps | 2-5 | Standard TCP/IP |
| WiFi (802.11ac) | 100m indoor | 5GHz | 433 Mbps | 3-8 | Standard TCP/IP |
| Bluetooth 5.0 | 240m | 2.4GHz | 2 Mbps | 0.1-0.5 | bluetooth_ros |
| Zigbee | 10-100m | 2.4GHz | 250 kbps | 0.5-1 | zigbee_ros |
| LoRa | 15km | 868/915MHz | 5.5 kbps | 0.5-2 | lora_ros |

### Fieldbus Protocols

| Protocol | Speed | Distance | Nodes | ROS Support | Use Case |
|----------|-------|----------|-------|-------------|----------|
| CAN | 1 Mbit/s | 40m | 110 | socketcan_interface | Motor control |
| EtherCAT | 100 Mbit/s | 100m | 65535 | soem | High-speed control |
| PROFINET | 100 Mbit/s | 100m | 256 | open62541 | Industrial |

## Pros and Cons of Common Platforms

### NVIDIA Jetson AGX Xavier
**Pros:**
- High-performance GPU for AI tasks
- Full ROS 2 compatibility
- Multiple connectivity options
- Good power efficiency

**Cons:**
- Expensive
- Requires good cooling
- Complex thermal management

### Intel RealSense D435
**Pros:**
- Integrated IMU
- Good depth accuracy
- ROS 2 support
- Compact form factor

**Cons:**
- Sensitive to sunlight
- Limited range
- Requires good lighting

### Dynamixel Servos
**Pros:**
- High precision control
- Feedback capabilities
- Easy to daisy-chain
- Good ROS support

**Cons:**
- Expensive for bulk purchase
- Requires specific controllers
- Limited to specific applications

## Cost Ranges

| Component Type | Budget Range | Mid-Range | High-End |
|----------------|--------------|-----------|----------|
| Computing | $100-300 | $300-800 | $800-2000 |
| Cameras | $50-200 | $200-800 | $800-2000 |
| LiDAR | $300-1000 | $1000-3000 | $3000+ |
| Actuators | $50-200 | $200-800 | $800+ |
| Complete Platforms | $1000-5000 | $5000-20000 | $20000+ |

## Installation Requirements

### Minimum Specifications
- CPU: Quad-core ARM or x86 processor
- RAM: 4GB minimum, 8GB recommended
- Storage: 32GB minimum, 64GB recommended
- Power: 12V DC with appropriate current rating
- Connectivity: WiFi or Ethernet for communication

### Recommended Specifications
- CPU: Hexa-core or octa-core processor
- RAM: 8GB minimum, 16GB recommended
- Storage: 64GB minimum, 128GB recommended
- Power: Regulated 12V DC with 20% overhead
- Connectivity: Dual-band WiFi + Ethernet backup

## Professional Development Considerations

### Integration Guidelines

When implementing robotics solutions for professional applications, consider these guidelines:

1. **System Reliability**: Implement redundant systems and error recovery mechanisms
2. **Safety Protocols**: Include emergency stops and collision avoidance systems
3. **Scalability**: Design systems that can be extended or modified
4. **Maintainability**: Use modular designs and comprehensive documentation
5. **Security**: Implement authentication and encryption for network communications

### Best Practices for Professional Deployment

**Hardware Selection:**
- Choose components with long-term availability and support
- Consider environmental factors (temperature, humidity, vibration)
- Plan for maintenance access and component replacement
- Include appropriate enclosures and protection

**Software Architecture:**
- Use established frameworks (ROS 2) for standardization
- Implement proper logging and monitoring
- Design for fault tolerance and graceful degradation
- Include comprehensive testing and validation procedures

### Common Professional Applications

| Application | Key Requirements | Recommended Platforms |
|-------------|------------------|----------------------|
| Industrial Automation | High precision, reliability, safety | UR5e, ABB IRC5, KUKA KR QUANTEC |
| Service Robotics | Human interaction, navigation, perception | TIAGo, Pepper, Aido |
| Research Platforms | Flexibility, extensibility, sensor integration | PR2, Fetch, TurtleBot 4 |
| Healthcare Assistance | Safety, hygiene, gentle interaction | RIBA, Robear, Moxi |
| Logistics/Delivery | Endurance, navigation, payload capacity | MiR100, Locus Robotics, Amazon Pegasus |

### Development Workflow for Professional Solutions

1. **Requirements Analysis**: Define functional and non-functional requirements
2. **System Architecture**: Design high-level system components and interfaces
3. **Component Selection**: Choose appropriate hardware and software components
4. **Implementation**: Develop and integrate system components
5. **Testing**: Perform unit, integration, and system-level testing
6. **Validation**: Verify system meets requirements in target environment
7. **Deployment**: Install and commission the system
8. **Maintenance**: Ongoing support, updates, and improvements

### Troubleshooting and Debugging

For professional robotics solutions, implement comprehensive debugging capabilities:

- **Remote Access**: Enable secure remote monitoring and control
- **Diagnostics**: Include system health monitoring and diagnostic tools
- **Logging**: Implement structured logging for debugging and analysis
- **Error Handling**: Design robust error detection and recovery procedures
- **Performance Monitoring**: Track system performance metrics