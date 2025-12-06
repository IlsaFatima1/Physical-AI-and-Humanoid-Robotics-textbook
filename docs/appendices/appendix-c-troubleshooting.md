# Appendix C: Troubleshooting Guide

This appendix provides solutions to common problems encountered when working with Physical AI and Humanoid Robotics systems, particularly those using ROS 2, Gazebo, and NVIDIA Isaac.

## ROS 2 Troubleshooting

### Installation Issues

**Problem**: `rosdep init` fails with permission error
- **Solution**: Run with sudo: `sudo rosdep init`
- **Alternative**: Check if rosdep is already initialized with `rosdep update`

**Problem**: `apt install` fails due to missing keys
- **Solution**:
  ```bash
  curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
  sudo apt update
  ```

**Problem**: ROS 2 environment not found
- **Solution**: Source the setup script: `source /opt/ros/humble/setup.bash`
- **Permanent fix**: Add the source command to your `~/.bashrc`

### Runtime Issues

**Problem**: Nodes cannot communicate across machines
- **Solution**: Set ROS domain ID and ensure network configuration:
  ```bash
  export ROS_DOMAIN_ID=0
  export ROS_LOCALHOST_ONLY=0  # For multi-machine setup
  ```

**Problem**: Nodes cannot find custom message types
- **Solution**:
  1. Ensure package is built: `colcon build --packages-select <package_name>`
  2. Source the workspace: `source install/setup.bash`
  3. Check message installation: `ros2 interface show <package_name>/msg/<MessageName>`

**Problem**: High latency in topic communication
- **Solution**:
  1. Check network configuration
  2. Reduce QoS settings for faster communication
  3. Use intra-process communication where possible

### Build Issues

**Problem**: `colcon build` fails with missing dependencies
- **Solution**: Install missing dependencies: `rosdep install --from-paths src --ignore-src -r -y`

**Problem**: CMake cannot find ROS packages
- **Solution**: Ensure `find_package()` calls match installed ROS packages:
  ```cmake
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  ```

## Gazebo Troubleshooting

### Installation and Setup

**Problem**: Gazebo fails to launch with graphics errors
- **Solution**: Check graphics drivers and run with software rendering:
  ```bash
  export LIBGL_ALWAYS_SOFTWARE=1
  gz sim
  ```

**Problem**: Models not loading in Gazebo
- **Solution**:
  1. Check GAZEBO_MODEL_PATH: `echo $GAZEBO_MODEL_PATH`
  2. Add custom models: `export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/path/to/models`
  3. Verify model structure follows Gazebo conventions

### Simulation Issues

**Problem**: Robot falls through the ground
- **Solution**:
  1. Check collision and visual elements in URDF/SDF
  2. Verify physics parameters (mass, inertia)
  3. Check for proper joint constraints

**Problem**: Physics simulation is unstable
- **Solution**:
  1. Reduce time step in physics configuration
  2. Adjust solver parameters (iterations, tolerance)
  3. Verify mass and inertia properties

**Problem**: Camera sensors not publishing images
- **Solution**:
  1. Check sensor plugin configuration
  2. Verify topic names and connections
  3. Check for proper rendering engine setup

## NVIDIA Isaac Troubleshooting

### Docker and Container Issues

**Problem**: Isaac Docker containers fail to start
- **Solution**:
  1. Verify NVIDIA drivers: `nvidia-smi`
  2. Check nvidia-docker installation: `docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi`
  3. Restart Docker: `sudo systemctl restart docker`

**Problem**: GPU acceleration not working in Isaac containers
- **Solution**:
  1. Run container with GPU access: `docker run --gpus all ...`
  2. Check CUDA installation inside container: `nvidia-smi`
  3. Verify Isaac packages are GPU-enabled

### Perception System Issues

**Problem**: Isaac perception nodes not processing data
- **Solution**:
  1. Verify input topics are publishing: `ros2 topic echo <topic_name>`
  2. Check Isaac node configuration
  3. Verify GPU memory availability

**Problem**: High latency in Isaac perception pipeline
- **Solution**:
  1. Reduce input data rate
  2. Optimize CUDA memory management
  3. Use appropriate model sizes for target hardware

## Humanoid Robot Specific Issues

### Balance and Stability

**Problem**: Humanoid robot is unstable or falls frequently
- **Solution**:
  1. Check center of mass calculation
  2. Verify IMU calibration and mounting
  3. Adjust balance controller gains
  4. Check joint compliance settings

**Problem**: Walking gait is unsteady
- **Solution**:
  1. Verify step parameters (height, duration, length)
  2. Check foot placement accuracy
  3. Adjust ZMP (Zero Moment Point) controller
  4. Verify sensor feedback timing

### Manipulation Issues

**Problem**: Robot cannot grasp objects reliably
- **Solution**:
  1. Calibrate end effectors and grippers
  2. Check grasp planning parameters
  3. Verify force/torque sensor calibration
  4. Adjust grasp force settings

**Problem**: Arm trajectory planning fails frequently
- **Solution**:
  1. Check joint limits and constraints
  2. Verify collision environment
  3. Adjust planning algorithm parameters
  4. Check for proper IK solver configuration

## Performance Optimization

### Memory Issues

**Problem**: System runs out of memory during operation
- **Solution**:
  1. Monitor memory usage: `htop` or `free -h`
  2. Reduce data rates for high-bandwidth topics
  3. Implement proper memory management in nodes
  4. Use memory pools for frequently allocated objects

### CPU Usage

**Problem**: High CPU usage causing performance issues
- **Solution**:
  1. Profile nodes to identify bottlenecks: `ros2 run plotjuggler plotjuggler`
  2. Optimize algorithms and reduce unnecessary computations
  3. Adjust node update rates
  4. Use multi-threading where appropriate

### Real-time Performance

**Problem**: Missed deadlines in real-time control
- **Solution**:
  1. Use real-time kernel if available
  2. Increase node priority: `chrt -f 99 <process>`
  3. Optimize control loop timing
  4. Reduce communication overhead

## Network and Communication Issues

### ROS 2 Network

**Problem**: Nodes cannot communicate across network
- **Solution**:
  1. Check firewall settings for ROS ports (11000+)
  2. Set appropriate ROS_DOMAIN_ID
  3. Verify network interface configuration
  4. Use Fast DDS configuration for better network performance

**Problem**: Intermittent connection drops
- **Solution**:
  1. Check network stability and latency
  2. Increase timeout values in QoS settings
  3. Implement reconnection logic in nodes
  4. Use reliable QoS policies for critical topics

### Sensor Communication

**Problem**: Sensor data not publishing consistently
- **Solution**:
  1. Check sensor hardware connections
  2. Verify appropriate update rates
  3. Check for buffer overflows
  4. Implement proper error handling

## Development Environment

### IDE and Tooling

**Problem**: IDE cannot find ROS 2 packages
- **Solution**:
  1. Install ROS extension for VS Code
  2. Source workspace in IDE terminal
  3. Configure CMake tools properly
  4. Set appropriate environment variables

**Problem**: Debugging tools not working with ROS 2
- **Solution**:
  1. Use ROS 2 debugging tools: `ros2 run rqt_plot rqt_plot`
  2. Implement proper logging: `RCLCPP_INFO(get_logger(), ...)`
  3. Use ROS 2 introspection tools: `ros2 topic info`, `ros2 node info`

### Version Control

**Problem**: Large binary files in Git repository
- **Solution**:
  1. Use Git LFS for large files: `git lfs track "*.dae" "*.stl"`
  2. Implement proper `.gitignore` for build artifacts
  3. Use submodules for external dependencies

## Common Error Messages and Solutions

### ROS 2 Errors

**Error**: `command not found: ros2`
- **Solution**: Source ROS 2 setup: `source /opt/ros/humble/setup.bash`

**Error**: `No executable found`
- **Solution**: Build packages: `colcon build` and source: `source install/setup.bash`

**Error**: `Could not find a package configuration file`
- **Solution**: Check package.xml dependencies and install missing packages

### Gazebo Errors

**Error**: `gz: command not found`
- **Solution**: Install Gazebo Garden and ensure it's in PATH

**Error**: `Failed to load the Gazebo ROS interface`
- **Solution**: Check ROS plugin configuration in SDF/URDF files

### Isaac Errors

**Error**: `CUDA error: out of memory`
- **Solution**: Reduce batch sizes or use smaller models

**Error**: `Isaac ROS node failed to initialize`
- **Solution**: Check Isaac package installation and GPU compatibility

## System Health Monitoring

### Process Monitoring
```bash
# Monitor ROS 2 processes
ros2 run top top

# Check system resources
htop
nvidia-smi  # For GPU monitoring
```

### Network Monitoring
```bash
# Check ROS 2 topics
ros2 topic list
ros2 topic hz <topic_name>  # Check topic frequency

# Check ROS 2 nodes
ros2 node list
ros2 lifecycle list <node_name>
```

### Log Analysis
```bash
# ROS 2 logging
ros2 run rqt_logger_level rqt_logger_level
ros2 param set <node_name> use_sim_time true  # For simulation time
```

## Preventive Maintenance

### Regular Checks

1. **System Updates**: Regularly update ROS 2, Gazebo, and Isaac packages
2. **Hardware Calibration**: Periodically calibrate sensors and actuators
3. **Backup Configurations**: Maintain backups of working configurations
4. **Documentation**: Keep troubleshooting notes and solutions updated

### Performance Baselines

Establish baseline performance metrics:
- CPU and memory usage under normal operation
- Topic communication rates and latencies
- Sensor data quality and consistency
- Control loop timing and accuracy

This troubleshooting guide should help resolve most common issues encountered in Physical AI and Humanoid Robotics development. When encountering new problems, consider the system architecture and check each component in the data flow from sensors to actuators.