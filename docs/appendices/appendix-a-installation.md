# Appendix A: Installation Guide

This appendix provides detailed instructions for installing the required software and tools for working with the Physical AI & Humanoid Robotics textbook.

## ROS 2 Humble Hawksbill Installation

### Ubuntu 22.04 LTS (Recommended)

1. Set locale:
   ```bash
   locale-gen en_US.UTF-8
   ```

2. Add the ROS 2 apt repository:
   ```bash
   apt update && apt install curl gnupg lsb-release
   curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
   ```

3. Install ROS 2 packages:
   ```bash
   apt update
   apt install ros-humble-desktop
   ```

4. Install colcon build tools:
   ```bash
   apt install python3-colcon-common-extensions
   ```

5. Source the ROS 2 environment:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

### Setting up the ROS 2 Environment

Add the following to your `~/.bashrc` file to automatically source ROS 2:
```bash
source /opt/ros/humble/setup.bash
```

## Gazebo Garden Installation

### Ubuntu 22.04 LTS

1. Add the Gazebo repository:
   ```bash
   curl -sSL http://get.gazebosim.org | sh
   ```

2. Install Gazebo Garden:
   ```bash
   apt install gz-garden
   ```

## NVIDIA Isaac ROS Installation

### Prerequisites

1. NVIDIA GPU with CUDA support
2. NVIDIA drivers installed
3. CUDA toolkit installed

### Installation Steps

1. Add the Isaac ROS apt repository:
   ```bash
   curl -sSL https://repo.rds.nvidia.com/ | sudo apt-key add -
   echo "deb https://repo.rds.nvidia.com/ubuntu/$(lsb_release -cs)/ arm64/" | sudo tee /etc/apt/sources.list.d/nvidia-isaac-ros.list
   apt update
   ```

2. Install Isaac ROS packages:
   ```bash
   apt install nvidia-isaac-ros-common nvidia-isaac-ros-core
   ```

## Python Environment Setup

### Create a Virtual Environment

```bash
python3 -m venv ~/ros2_env
source ~/ros2_env/bin/activate
pip install --upgrade pip
```

### Install Required Python Packages

```bash
pip install numpy matplotlib scipy
```

## Verification Steps

After installation, verify that the systems are working:

1. Test ROS 2:
   ```bash
   source /opt/ros/humble/setup.bash
   ros2 topic list
   ```

2. Test Gazebo:
   ```bash
   gz sim
   ```

3. Test Python environment:
   ```bash
   python3 -c "import numpy; print('NumPy version:', numpy.__version__)"
   ```

## Advanced Installation Options

### Docker-based Installation
For isolated development environments, consider using Docker:

1. Install Docker:
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   sudo usermod -aG docker $USER
   ```

2. Create a ROS 2 development container:
   ```bash
   docker run -it --rm \
     --name ros2_dev \
     --env DISPLAY=$DISPLAY \
     --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
     --volume $HOME/ros2_workspace:/workspace \
     osrf/ros:humble-desktop
   ```

### Source-based Installation
For the latest features or development work:

1. Install build dependencies:
   ```bash
   sudo apt update
   sudo apt install -y \
     build-essential \
     cmake \
     git \
     python3-colcon-common-extensions \
     python3-rosdep \
     python3-vcstool \
     wget
   ```

2. Create a colcon workspace:
   ```bash
   mkdir -p ~/ros2_src/src
   cd ~/ros2_src
   ```

3. Get ROS 2 source code:
   ```bash
   wget https://raw.githubusercontent.com/ros2/ros2/humble/ros2.repos
   vcs import src < ros2.repos
   ```

4. Install dependencies:
   ```bash
   rosdep update
   rosdep install --from-paths src --ignore-src -y --skip-keys "libopencv-dev libopencv-contrib-dev libopencv-imgproc-dev python3-opencv"
   ```

5. Build the workspace:
   ```bash
   colcon build --symlink-install
   source install/setup.bash
   ```

## Troubleshooting Common Installation Issues

### Permission Issues
If you encounter permission issues, make sure you're running commands with appropriate privileges or have set up proper user groups (like dialout for serial communication).

### Network Issues
Some installations may require proxy settings. If behind a corporate firewall, configure your apt proxy settings:
```bash
export http_proxy=http://proxy.company.com:port
export https_proxy=http://proxy.company.com:port
```

### Missing Dependencies
If installation fails due to missing dependencies, run:
```bash
apt update
apt install -f
```

### Gazebo Installation Issues
If Gazebo fails to install or launch:
1. Check graphics drivers: `lspci | grep -i vga`
2. Install graphics libraries: `sudo apt install nvidia-prime mesa-utils`
3. Test OpenGL: `glxinfo | grep "OpenGL renderer"`

### Isaac ROS Installation Issues
For Isaac-specific problems:
1. Verify CUDA installation: `nvcc --version`
2. Check GPU compatibility: `nvidia-smi`
3. Install Isaac dependencies: `sudo apt install nvidia-isaac-*`

## System Requirements Verification

### Hardware Requirements Check
```bash
# Check CPU
lscpu

# Check memory
free -h

# Check disk space
df -h

# Check GPU (if applicable)
nvidia-smi  # For NVIDIA GPUs
```

### Software Requirements Check
```bash
# Check Ubuntu version
lsb_release -a

# Check GCC version
gcc --version

# Check CMake version
cmake --version

# Check Python version
python3 --version
```

## Post-Installation Configuration

### Environment Setup
Add these lines to your `~/.bashrc` for persistent ROS 2 setup:
```bash
# ROS 2 Humble Setup
source /opt/ros/humble/setup.bash

# Custom workspace setup (if you created one)
# source ~/ros2_ws/install/setup.bash

# Gazebo setup
source /usr/share/gazebo/setup.sh

# Custom aliases for development
alias cb='cd ~/ros2_ws && colcon build'
alias sb='source ~/ros2_ws/install/setup.bash'
alias gs='gz sim'
```

### Network Configuration for Multi-Robot Systems
For networked robotics applications:
```bash
# Set ROS domain ID (0-255)
export ROS_DOMAIN_ID=0

# For localhost-only communication (simulation)
export ROS_LOCALHOST_ONLY=1

# For multi-machine communication
export ROS_LOCALHOST_ONLY=0
```

### Performance Tuning
For better real-time performance:
```bash
# Increase shared memory size
echo "kernel.shmmax=134217728" | sudo tee -a /etc/sysctl.conf

# Increase file descriptor limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
```

## Development Environment Setup

### IDE Configuration
For VS Code with ROS 2 development:

1. Install the ROS extension pack
2. Create a workspace file `.code-workspace`:
```json
{
    "folders": [
        {
            "path": "."
        }
    ],
    "settings": {
        "python.defaultInterpreterPath": "~/ros2_env/bin/python",
        "cmake.cmakePath": "/usr/bin/cmake",
        "terminal.integrated.env.bash": {
            "CMAKE_PREFIX_PATH": "/opt/ros/humble",
            "AMENT_PREFIX_PATH": "/opt/ros/humble",
            "LD_LIBRARY_PATH": "/opt/ros/humble/lib"
        }
    }
}
```

### Development Tools Installation
```bash
# Install development tools
sudo apt install -y \
  git \
  vim \
  tmux \
  htop \
  iotop \
  iftop \
  build-essential \
  gdb \
  valgrind

# Install Python development tools
pip3 install \
  flake8 \
  black \
  isort \
  pytest \
  pytest-cov
```

## Verification and Testing

### Basic ROS 2 Test
```bash
# Terminal 1
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_cpp talker

# Terminal 2
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_py listener
```

### Gazebo Test
```bash
# Launch a simple world
gz sim -r empty.sdf

# Launch with GUI
gz sim -g -r shapes.sdf
```

### Isaac Test (if installed)
```bash
# Check Isaac packages
apt list --installed | grep nvidia-isaac

# Test basic functionality
# (Specific tests depend on installed Isaac packages)
```

This installation guide provides comprehensive instructions for setting up the Physical AI and Humanoid Robotics development environment, including basic, advanced, and troubleshooting sections to ensure successful setup for all users.