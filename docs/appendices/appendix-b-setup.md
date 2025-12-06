# Appendix B: Setup Guide

This appendix provides detailed setup procedures for configuring your development environment to work with the Physical AI & Humanoid Robotics textbook examples.

## ROS 2 Workspace Setup

### Create a New Workspace

1. Create the workspace directory:
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws
   ```

2. Source ROS 2:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

3. Build the workspace:
   ```bash
   colcon build
   ```

4. Source the workspace:
   ```bash
   source install/setup.bash
   ```

### Add Workspace to Environment

Add the following to your `~/.bashrc` to automatically source your workspace:
```bash
source ~/ros2_ws/install/setup.bash
```

## Gazebo Configuration

### Environment Variables

Set the following environment variables for Gazebo:
```bash
export GZ_SIM_RESOURCE_PATH=$HOME/.gazebo/models
export GZ_SIM_SYSTEM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gazebo-11/plugins
export GZ_SIM_MEDIA_PATH=$HOME/.gazebo
```

### Model Path Configuration

Add custom model paths to your `~/.gazebo/setup.sh`:
```bash
export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:/path/to/your/models
```

## Isaac ROS Configuration

### Docker Setup

Isaac ROS often uses Docker containers for isolation. Install Docker:
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER
```

### NVIDIA Container Runtime

Install nvidia-docker for GPU support:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Development Environment Setup

### IDE Configuration

For VS Code with ROS 2 development:

1. Install the ROS extension pack
2. Configure the workspace with the following settings in `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "~/ros2_env/bin/python",
    "cmake.cmakePath": "/usr/bin/cmake",
    "terminal.integrated.env.bash": {
        "CMAKE_PREFIX_PATH": "/opt/ros/humble",
        "AMENT_PREFIX_PATH": "/opt/ros/humble",
        "LD_LIBRARY_PATH": "/opt/ros/humble/lib"
    }
}
```

### ROS 2 Development Tools

Install additional development tools:
```bash
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

## Simulation Environment Setup

### Create a Simulation Workspace

1. Create a simulation-specific workspace:
   ```bash
   mkdir -p ~/simulation_ws/src
   cd ~/simulation_ws
   ```

2. Create a simulation launch package:
   ```bash
   cd src
   ros2 pkg create --build-type ament_python simulation_launch
   ```

3. Build the workspace:
   ```bash
   cd ~/simulation_ws
   colcon build
   source install/setup.bash
   ```

## Hardware Interface Setup

### Serial Communication

For direct hardware communication, add your user to the dialout group:
```bash
sudo usermod -a -G dialout $USER
```

### Network Configuration

For network-based hardware interfaces:
```bash
# Check network interfaces
ip addr show

# Set up static IP if needed
sudo nano /etc/netplan/01-netcfg.yaml
```

## Testing Your Setup

### Basic ROS 2 Test

1. Open two terminals
2. In the first terminal, source ROS 2 and run a talker:
   ```bash
   source /opt/ros/humble/setup.bash
   ros2 run demo_nodes_cpp talker
   ```

3. In the second terminal, source ROS 2 and run a listener:
   ```bash
   source /opt/ros/humble/setup.bash
   ros2 run demo_nodes_py listener
   ```

### Gazebo Test

Launch a simple Gazebo world:
```bash
gz sim -r empty.sdf
```

### Python Environment Test

Test Python packages:
```bash
python3 -c "
import rclpy
import numpy as np
import matplotlib.pyplot as plt
print('All required packages are available')
"
```

## Troubleshooting Setup Issues

### Common Issues and Solutions

1. **Package Not Found**: Make sure you've sourced your workspace:
   ```bash
   source ~/ros2_ws/install/setup.bash
   ```

2. **Permission Denied**: Check file permissions and ensure you're in the correct groups:
   ```bash
   groups $USER
   ```

3. **Library Not Found**: Add library paths to your environment:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
   ```

4. **Port Already in Use**: Check for running processes:
   ```bash
   lsof -i :port_number
   ```

## Advanced Setup Configurations

### Real-time Configuration for Humanoid Robots

For humanoid robots requiring precise timing, configure real-time capabilities:

1. **Install real-time kernel** (if available for your hardware):
   ```bash
   sudo apt install linux-headers-$(uname -r)-rt
   ```

2. **Configure CPU isolation** for real-time performance:
   ```bash
   # Add to GRUB configuration in /etc/default/grub
   GRUB_CMDLINE_LINUX="isolcpus=1,2,3"
   sudo update-grub
   ```

3. **Set real-time permissions**:
   ```bash
   # Add to /etc/security/limits.conf
   * soft rtprio 99
   * hard rtprio 99
   ```

### Simulation-Specific Setup

For development using Gazebo simulation:

1. **Optimize graphics performance**:
   ```bash
   # For NVIDIA GPUs
   nvidia-smi -pm 1
   nvidia-smi -ac 2505,875  # Adjust for your GPU
   ```

2. **Configure Gazebo environment variables**:
   ```bash
   export GZ_SIM_RESOURCE_PATH=$HOME/.gazebo/models:/usr/share/gazebo-11/models
   export GZ_SIM_SYSTEM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/gazebo-11/plugins
   export GZ_SIM_MEDIA_PATH=$HOME/.gazebo
   ```

### Isaac-Specific Setup

For NVIDIA Isaac development:

1. **Verify CUDA and cuDNN installation**:
   ```bash
   nvcc --version
   cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
   ```

2. **Configure Isaac environment**:
   ```bash
   export ISAAC_ROS_WS=$HOME/isaac_ros_ws
   source $ISAAC_ROS_WS/install/setup.bash
   ```

3. **Set GPU memory configuration**:
   ```bash
   # For Isaac applications requiring large GPU memory
   export CUDA_DEVICE_ORDER=PCI_BUS_ID
   export CUDA_VISIBLE_DEVICES=0
   ```

## Development Workflow Setup

### Git Configuration for Robotics Projects

Configure Git for robotics development with large binary files:

1. **Install Git LFS** for large assets:
   ```bash
   git lfs install
   git lfs track "*.dae" "*.stl" "*.obj" "*.urdf" "*.sdf"
   ```

2. **Configure Git for robotics projects**:
   ```bash
   git config --global core.precomposeunicode true
   git config --global pull.rebase false
   git config --global init.defaultBranch main
   ```

### Workspace Organization

Recommended workspace structure for robotics projects:

```
~/ros2_ws/
├── src/
│   ├── robot_packages/
│   │   ├── robot_description/
│   │   ├── robot_control/
│   │   └── robot_perception/
│   ├── common_packages/
│   │   ├── common_msgs/
│   │   └── common_utils/
│   └── external_packages/
│       ├── navigation2/
│       └── perception/
├── install/
├── build/
└── log/
```

### Build System Optimization

For faster builds in robotics projects:

1. **Configure colcon for optimal performance**:
   ```bash
   # Create colcon configuration
   mkdir -p ~/.colcon
   cat > ~/.colcon/defaults.yaml << EOF
   {
       "build": {
           "cmake-args": [
               "-DCMAKE_BUILD_TYPE=Release",
               "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
           ],
           "parallel-workers": 8
       }
   }
   EOF
   ```

2. **Use colcon with specific package building**:
   ```bash
   # Build only specific packages
   colcon build --packages-select robot_control robot_perception

   # Build with symlinks to speed up rebuilds
   colcon build --symlink-install
   ```

## Hardware Interface Setup

### USB Device Permissions

For robots with USB-connected components:

1. **Create udev rules** for consistent device naming:
   ```bash
   # Create rule file
   sudo tee /etc/udev/rules.d/99-robot-usb.rules << EOF
   SUBSYSTEM=="tty", ATTRS{idVendor}=="1234", ATTRS{idProduct}=="5678", SYMLINK+="robot_controller"
   EOF

   # Reload rules
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```

2. **Add user to dialout group** for serial communication:
   ```bash
   sudo usermod -a -G dialout $USER
   # Log out and back in for changes to take effect
   ```

### Network Configuration for Distributed Robotics

For multi-robot or distributed systems:

1. **Configure ROS network**:
   ```bash
   # Set up ROS network in ~/.bashrc
   export ROS_HOSTNAME=$(hostname).local
   export ROS_MASTER_URI=http://$(hostname).local:11311
   export ROS_IP=$(hostname -I | awk '{print $1}')
   ```

2. **Set up multicast DNS** for easy hostname resolution:
   ```bash
   sudo apt install avahi-daemon avahi-utils
   sudo systemctl enable avahi-daemon
   ```

## Performance Monitoring Setup

### System Monitoring Tools

Install and configure monitoring tools for robotics applications:

```bash
# Install monitoring tools
sudo apt install htop iotop iftop nethogs ncdu

# Install ROS-specific monitoring
sudo apt install ros-humble-rqt ros-humble-rqt-top ros-humble-rqt-plot

# Install GPU monitoring (for Isaac systems)
sudo apt install nvidia-ml-dev
```

### Resource Usage Baselines

Establish baseline measurements for your robot system:

1. **Idle system resource usage**:
   ```bash
   # Record baseline CPU, memory, and network usage
   top -b -n 1 > baseline_system_usage.txt
   free -h >> baseline_system_usage.txt
   iftop -t -s 10 -L 10 >> baseline_network_usage.txt
   ```

2. **ROS 2 system monitoring**:
   ```bash
   # Monitor ROS 2 topics and their frequencies
   ros2 topic list
   ros2 run plotjuggler plotjuggler  # For real-time visualization
   ```

## Backup and Recovery Setup

### Configuration Backup

Set up regular backups of important configurations:

```bash
# Create backup script
cat > ~/backup_robot_configs.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="$HOME/robot_backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup configurations
cp -r ~/.bashrc ~/.profile ~/ros2_ws/src/*/config "$BACKUP_DIR/" 2>/dev/null || true

# Backup calibration files
find ~/ros2_ws/src -name "*calibration*" -exec cp --parents {} "$BACKUP_DIR/" \; 2>/dev/null || true

# Create system information file
uname -a > "$BACKUP_DIR/system_info.txt"
lscpu >> "$BACKUP_DIR/system_info.txt"
free -h >> "$BACKUP_DIR/system_info.txt"

echo "Backup completed to $BACKUP_DIR"
EOF

chmod +x ~/backup_robot_configs.sh
```

### Version Control Strategy

Implement a comprehensive version control strategy:

1. **Separate repositories** for different components:
   - Robot hardware description
   - Robot software packages
   - Simulation environments
   - Calibration files

2. **Git hooks** for code quality:
   ```bash
   # Install pre-commit hooks
   pip3 install pre-commit
   pre-commit sample > .pre-commit-config.yaml
   pre-commit install
   ```

This setup guide provides comprehensive instructions for configuring your development environment for Physical AI and Humanoid Robotics, including advanced configurations for real-time systems, simulation, and hardware integration.