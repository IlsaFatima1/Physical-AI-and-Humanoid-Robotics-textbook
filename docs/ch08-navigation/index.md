# Chapter 8: Mobile Robot Navigation and Path Planning

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the fundamental concepts of mobile robot navigation and path planning
- Implement classical and modern path planning algorithms
- Integrate navigation systems with ROS 2 and perception modules
- Design behavior-based navigation systems for dynamic environments
- Apply motion planning techniques for obstacle avoidance and trajectory generation
- Evaluate navigation system performance and handle uncertainty in real-world scenarios

## 8.1 Introduction to Mobile Robot Navigation

Mobile robot navigation is the process by which a robot determines how to move from its current location to a desired goal while avoiding obstacles and respecting environmental constraints. Navigation systems are fundamental to autonomous robotics, enabling robots to operate effectively in unstructured environments.

### 8.1.1 The Navigation Stack

Modern mobile robot navigation typically follows a hierarchical structure:

1. **Localization**: Determining the robot's position in the environment
2. **Mapping**: Creating or updating a representation of the environment
3. **Path Planning**: Computing a geometric path from start to goal
4. **Motion Planning**: Generating time-parameterized trajectories
5. **Control**: Executing the planned motions while handling real-time feedback

### 8.1.2 Navigation Challenges

Mobile robot navigation faces several key challenges:

- **Perception Uncertainty**: Sensor noise and limited field of view
- **Dynamic Environments**: Moving obstacles and changing conditions
- **Real-time Constraints**: Need for fast computation and response
- **Safety Requirements**: Avoiding collisions and ensuring system safety
- **Computational Complexity**: Balancing accuracy with efficiency

## 8.2 Localization and Mapping

### 8.2.1 Simultaneous Localization and Mapping (SLAM)

SLAM is the process of building a map of an unknown environment while simultaneously localizing the robot within that map. This is fundamental for navigation in previously unmapped environments.

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

class ParticleFilterSLAM:
    def __init__(self, num_particles=100):
        self.num_particles = num_particles
        self.particles = self.initialize_particles()
        self.map = {}  # Dictionary to store landmarks

    def initialize_particles(self):
        """Initialize particles with random poses"""
        particles = []
        for _ in range(self.num_particles):
            # Random pose (x, y, theta)
            pose = np.random.uniform(-10, 10, 3)
            weight = 1.0 / self.num_particles
            particles.append({'pose': pose, 'weight': weight, 'landmarks': {}})
        return particles

    def predict_motion(self, control_input, dt):
        """Predict robot motion using odometry model"""
        for particle in self.particles:
            # Add noise to control input
            noisy_control = control_input + np.random.normal(0, 0.1, 2)

            # Update pose based on control input
            x, y, theta = particle['pose']
            v, omega = noisy_control

            # Differential drive motion model
            if abs(omega) < 1e-6:  # Straight line motion
                x_new = x + v * dt * math.cos(theta)
                y_new = y + v * dt * math.sin(theta)
                theta_new = theta
            else:  # Circular motion
                radius = v / omega
                x_new = x + radius * (math.sin(theta + omega * dt) - math.sin(theta))
                y_new = y + radius * (math.cos(theta) - math.cos(theta + omega * dt))
                theta_new = theta + omega * dt

            particle['pose'] = np.array([x_new, y_new, theta_new])

    def update_weights(self, sensor_data):
        """Update particle weights based on sensor observations"""
        for particle in self.particles:
            weight = 1.0
            for obs in sensor_data:
                # Calculate expected observation based on particle's map
                expected_obs = self.calculate_expected_observation(particle, obs['landmark_id'])
                if expected_obs is not None:
                    # Calculate likelihood of observation
                    likelihood = self.calculate_observation_likelihood(obs, expected_obs)
                    weight *= likelihood

            particle['weight'] = weight

        # Normalize weights
        total_weight = sum(p['weight'] for p in self.particles)
        if total_weight > 0:
            for particle in self.particles:
                particle['weight'] /= total_weight

    def calculate_expected_observation(self, particle, landmark_id):
        """Calculate expected sensor observation for a landmark"""
        if landmark_id in particle['landmarks']:
            landmark_pos = particle['landmarks'][landmark_id]
            robot_x, robot_y, robot_theta = particle['pose']

            # Calculate expected range and bearing
            dx = landmark_pos[0] - robot_x
            dy = landmark_pos[1] - robot_y
            expected_range = math.sqrt(dx*dx + dy*dy)
            expected_bearing = math.atan2(dy, dx) - robot_theta

            return {'range': expected_range, 'bearing': expected_bearing}
        return None

    def calculate_observation_likelihood(self, observed, expected):
        """Calculate likelihood of observation given expected value"""
        if expected is None:
            return 0.01  # Low likelihood if landmark not in map

        range_diff = observed['range'] - expected['range']
        bearing_diff = observed['bearing'] - expected['bearing']

        # Normalize bearing difference
        bearing_diff = (bearing_diff + math.pi) % (2 * math.pi) - math.pi

        # Gaussian likelihood
        range_likelihood = math.exp(-0.5 * (range_diff / 0.5)**2)
        bearing_likelihood = math.exp(-0.5 * (bearing_diff / 0.1)**2)

        return range_likelihood * bearing_likelihood

    def resample(self):
        """Resample particles based on their weights"""
        weights = [p['weight'] for p in self.particles]
        indices = np.random.choice(len(self.particles), size=self.num_particles, p=weights)

        new_particles = []
        for idx in indices:
            new_particles.append(self.particles[idx].copy())

        self.particles = new_particles

        # Reset weights
        for particle in self.particles:
            particle['weight'] = 1.0 / self.num_particles
```

### 8.2.2 Occupancy Grid Mapping

Occupancy grid mapping represents the environment as a 2D grid where each cell contains the probability of being occupied:

```python
class OccupancyGridMap:
    def __init__(self, width, height, resolution=0.1):
        self.width = width
        self.height = height
        self.resolution = resolution  # meters per cell
        self.grid = np.full((height, width), 0.5)  # Initialize to unknown (0.5)

        # Log odds representation for efficient updates
        self.log_odds = np.zeros((height, width))

        # Conversion factors
        self.origin_x = 0.0
        self.origin_y = 0.0

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        x = grid_x * self.resolution + self.origin_x
        y = grid_y * self.resolution + self.origin_y
        return x, y

    def update_with_laser_scan(self, robot_pose, scan_ranges, scan_angles):
        """Update grid based on laser scan data"""
        robot_x, robot_y, robot_theta = robot_pose

        # Convert scan to world coordinates
        for i, (range_val, angle) in enumerate(zip(scan_ranges, scan_angles)):
            if range_val < 0.1 or range_val > 30.0:  # Invalid range
                continue

            # Calculate endpoint of this ray
            world_x = robot_x + range_val * math.cos(robot_theta + angle)
            world_y = robot_y + range_val * math.sin(robot_theta + angle)

            # Get grid coordinates
            end_grid_x, end_grid_y = self.world_to_grid(world_x, world_y)
            start_grid_x, start_grid_y = self.world_to_grid(robot_x, robot_y)

            # Perform ray tracing to update grid
            self.ray_trace(start_grid_x, start_grid_y, end_grid_x, end_grid_y, range_val < 29.0)  # endpoint is obstacle if not max range

    def ray_trace(self, x0, y0, x1, y1, endpoint_occupied):
        """Perform ray tracing between two points"""
        # Bresenham's line algorithm with probability updates
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x_step = 1 if x0 < x1 else -1
        y_step = 1 if y0 < y1 else -1

        error = dx - dy
        x, y = x0, y0

        # Update cells along the ray
        while x != x1 or y != y1:
            if 0 <= x < self.width and 0 <= y < self.height:
                # Update log odds (free space)
                self.log_odds[y, x] += self.log_odds_from_probability(0.4)  # Free space

            if error * 2 > -dy:
                error -= dy
                x += x_step
            if error * 2 < dx:
                error += dx
                y += y_step

        # Update endpoint
        if 0 <= x1 < self.width and 0 <= y1 < self.height:
            if endpoint_occupied:
                # Update log odds (occupied space)
                self.log_odds[y1, x1] += self.log_odds_from_probability(0.8)  # Occupied space
            else:
                # Update log odds (free space)
                self.log_odds[y1, x1] += self.log_odds_from_probability(0.3)  # Free space

    def log_odds_from_probability(self, prob):
        """Convert probability to log odds"""
        prob = max(0.01, min(0.99, prob))  # Clamp to avoid log(0)
        return math.log(prob / (1 - prob))

    def probability_from_log_odds(self, log_odds):
        """Convert log odds to probability"""
        prob = 1 - 1 / (1 + math.exp(log_odds))
        return prob

    def get_probability(self, x, y):
        """Get occupancy probability at world coordinates"""
        grid_x, grid_y = self.world_to_grid(x, y)
        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            return self.probability_from_log_odds(self.log_odds[grid_y, grid_x])
        return 0.5  # Unknown area
```

## 8.3 Path Planning Algorithms

### 8.3.1 A* Algorithm

A* is a popular graph search algorithm that finds the shortest path between start and goal while considering heuristic information:

```python
import heapq
from typing import List, Tuple

class AStarPlanner:
    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.height, self.width = grid_map.shape

    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Plan path using A* algorithm"""
        # Check if start and goal are valid
        if not self.is_valid_cell(start[0], start[1]) or not self.is_valid_cell(goal[0], goal[1]):
            return []

        # Initialize data structures
        open_set = [(0, start)]  # (f_score, position)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            # Explore neighbors
            for neighbor in self.get_neighbors(current[0], current[1]):
                tentative_g_score = g_score[current] + self.distance(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found

    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate heuristic distance (Euclidean)"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate distance between two adjacent cells"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighbors of a cell"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_valid_cell(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def is_valid_cell(self, x: int, y: int) -> bool:
        """Check if cell is valid and not occupied"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        # Assume grid_map contains occupancy probabilities (0.0 = free, 1.0 = occupied)
        return self.grid_map[y, x] < 0.7  # Threshold for considering cell as free

    def reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
```

### 8.3.2 Dijkstra's Algorithm

Dijkstra's algorithm finds the shortest path without using heuristic information:

```python
class DijkstraPlanner:
    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.height, self.width = grid_map.shape

    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Plan path using Dijkstra's algorithm"""
        if not self.is_valid_cell(start[0], start[1]) or not self.is_valid_cell(goal[0], goal[1]):
            return []

        # Initialize data structures
        open_set = [(0, start)]  # (distance, position)
        came_from = {}
        distances = {start: 0}

        while open_set:
            current_dist, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            # Explore neighbors
            for neighbor in self.get_neighbors(current[0], current[1]):
                new_dist = distances[current] + self.distance(current, neighbor)

                if neighbor not in distances or new_dist < distances[neighbor]:
                    came_from[neighbor] = current
                    distances[neighbor] = new_dist
                    heapq.heappush(open_set, (new_dist, neighbor))

        return []  # No path found

    def distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate distance between two adjacent cells"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighbors of a cell"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_valid_cell(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def is_valid_cell(self, x: int, y: int) -> bool:
        """Check if cell is valid and not occupied"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.grid_map[y, x] < 0.7

    def reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
```

### 8.3.3 RRT (Rapidly-exploring Random Trees)

RRT is a sampling-based algorithm suitable for high-dimensional configuration spaces:

```python
class RRTPlanner:
    def __init__(self, grid_map, step_size=1.0):
        self.grid_map = grid_map
        self.height, self.width = grid_map.shape
        self.step_size = step_size

    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float], max_iterations=10000) -> List[Tuple[float, float]]:
        """Plan path using RRT algorithm"""
        tree = [start]
        parent = {start: None}

        for _ in range(max_iterations):
            # Sample random point
            rand_point = self.sample_free_space()

            # Find nearest point in tree
            nearest = self.nearest_node(tree, rand_point)

            # Create new point in direction of random point
            new_point = self.steer(nearest, rand_point)

            # Check if path is collision-free
            if self.is_collision_free(nearest, new_point):
                tree.append(new_point)
                parent[new_point] = nearest

                # Check if we can connect to goal
                if self.distance(new_point, goal) < self.step_size:
                    if self.is_collision_free(new_point, goal):
                        parent[goal] = new_point
                        return self.reconstruct_path_rrt(parent, start, goal)

        return []  # No path found

    def sample_free_space(self) -> Tuple[float, float]:
        """Sample a random point in free space"""
        while True:
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)

            grid_x, grid_y = int(x), int(y)
            if (0 <= grid_x < self.width and 0 <= grid_y < self.height and
                self.grid_map[grid_y, grid_x] < 0.7):
                return (x, y)

    def nearest_node(self, tree: List[Tuple[float, float]], point: Tuple[float, float]) -> Tuple[float, float]:
        """Find the nearest node in the tree to the given point"""
        nearest = tree[0]
        min_dist = self.distance(nearest, point)

        for node in tree[1:]:
            dist = self.distance(node, point)
            if dist < min_dist:
                min_dist = dist
                nearest = node

        return nearest

    def steer(self, from_node: Tuple[float, float], to_node: Tuple[float, float]) -> Tuple[float, float]:
        """Create a new node in the direction from from_node to to_node"""
        dist = self.distance(from_node, to_node)
        if dist <= self.step_size:
            return to_node

        # Create point in direction of to_node at step_size distance
        direction = ((to_node[0] - from_node[0]) / dist,
                     (to_node[1] - from_node[1]) / dist)
        new_x = from_node[0] + self.step_size * direction[0]
        new_y = from_node[1] + self.step_size * direction[1]

        return (new_x, new_y)

    def is_collision_free(self, from_node: Tuple[float, float], to_node: Tuple[float, float]) -> bool:
        """Check if the path between two nodes is collision-free"""
        # Simple check: if both endpoints are free, assume path is free
        # In practice, you'd check multiple points along the path
        x1, y1 = from_node
        x2, y2 = to_node

        grid_x1, grid_y1 = int(x1), int(y1)
        grid_x2, grid_y2 = int(x2), int(y2)

        # Check both endpoints
        if (0 <= grid_x1 < self.width and 0 <= grid_y1 < self.height and
            self.grid_map[grid_y1, grid_x1] >= 0.7):
            return False

        if (0 <= grid_x2 < self.width and 0 <= grid_y2 < self.height and
            self.grid_map[grid_y2, grid_x2] >= 0.7):
            return False

        # For a more thorough check, sample points along the line
        steps = int(self.distance(from_node, to_node) / 0.5)  # Check every 0.5m
        for i in range(1, steps):
            t = i / steps
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)

            grid_x, grid_y = int(x), int(y)
            if (0 <= grid_x < self.width and 0 <= grid_y < self.height and
                self.grid_map[grid_y, grid_x] >= 0.7):
                return False

        return True

    def distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def reconstruct_path_rrt(self, parent: dict, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Reconstruct path from RRT tree"""
        path = [goal]
        current = goal

        while current != start:
            if current not in parent:
                return []  # No path found
            current = parent[current]
            path.append(current)

        path.reverse()
        return path
```

## 8.4 ROS 2 Navigation Integration

### 8.4.1 Navigation2 Stack Components

The Navigation2 stack provides a comprehensive navigation system for ROS 2:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose
import tf2_ros
import numpy as np
import math

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        # Initialize navigation components
        self.current_pose = None
        self.goal_pose = None
        self.map = None
        self.path = None
        self.laser_scan = None

        # TF listener for transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Setup subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, '/amcl_pose', self.pose_callback, 10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10
        )

        # Setup publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/plan', 10)

        # Setup navigation timer
        self.nav_timer = self.create_timer(0.1, self.navigation_loop)

        # Path planner
        self.path_planner = None
        self.local_planner = LocalPlanner()

        self.get_logger().info("Navigation node initialized")

    def pose_callback(self, msg):
        """Update current robot pose"""
        self.current_pose = msg.pose

    def map_callback(self, msg):
        """Update map data"""
        self.map = msg
        # Convert to numpy array for path planning
        self.map_array = np.array(msg.data).reshape(msg.info.height, msg.info.width)

    def scan_callback(self, msg):
        """Update laser scan data"""
        self.laser_scan = msg

    def goal_callback(self, msg):
        """Set navigation goal"""
        self.goal_pose = msg.pose
        self.get_logger().info(f"New goal set: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

        # Plan path to goal
        if self.current_pose and self.map:
            self.plan_path()

    def plan_path(self):
        """Plan path from current position to goal"""
        if not self.current_pose or not self.goal_pose or self.map is None:
            return

        # Convert poses to grid coordinates
        start_grid = self.world_to_grid(
            self.current_pose.position.x,
            self.current_pose.position.y
        )
        goal_grid = self.world_to_grid(
            self.goal_pose.position.x,
            self.goal_pose.position.y
        )

        # Create path planner
        self.path_planner = AStarPlanner(self.map_array)

        # Plan path
        path_grid = self.path_planner.plan_path(start_grid, goal_grid)

        if path_grid:
            # Convert grid path back to world coordinates
            self.path = []
            for grid_x, grid_y in path_grid:
                world_x, world_y = self.grid_to_world(grid_x, grid_y)
                pose_stamped = PoseStamped()
                pose_stamped.pose.position.x = world_x
                pose_stamped.pose.position.y = world_y
                pose_stamped.pose.orientation.w = 1.0  # No rotation
                self.path.append(pose_stamped)

            # Publish path
            path_msg = Path()
            path_msg.poses = self.path
            path_msg.header.frame_id = "map"
            path_msg.header.stamp = self.get_clock().now().to_msg()
            self.path_pub.publish(path_msg)

            self.get_logger().info(f"Path planned with {len(self.path)} waypoints")
        else:
            self.get_logger().warn("No path found to goal")

    def navigation_loop(self):
        """Main navigation control loop"""
        if self.current_pose and self.path and self.path:
            # Follow the path using local planner
            cmd_vel = self.local_planner.compute_velocity(
                self.current_pose,
                self.path,
                self.laser_scan
            )
            self.cmd_vel_pub.publish(cmd_vel)

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        if self.map is None:
            return (0, 0)

        grid_x = int((x - self.map.info.origin.position.x) / self.map.info.resolution)
        grid_y = int((y - self.map.info.origin.position.y) / self.map.info.resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        if self.map is None:
            return (0, 0)

        x = grid_x * self.map.info.resolution + self.map.info.origin.position.x
        y = grid_y * self.map.info.resolution + self.map.info.origin.position.y
        return (x, y)


class LocalPlanner:
    """Local planner for following global path and obstacle avoidance"""
    def __init__(self):
        self.lookahead_distance = 1.0
        self.max_linear_speed = 0.5
        self.max_angular_speed = 1.0

    def compute_velocity(self, current_pose, global_path, laser_scan):
        """Compute velocity command to follow path and avoid obstacles"""
        cmd_vel = Twist()

        if not global_path:
            return cmd_vel

        # Find closest point on path
        closest_idx = self.find_closest_waypoint(current_pose, global_path)

        # Find point to follow (lookahead)
        follow_idx = self.find_lookahead_point(current_pose, global_path, closest_idx)

        if follow_idx is not None:
            # Calculate desired direction
            target_x = global_path[follow_idx].pose.position.x
            target_y = global_path[follow_idx].pose.position.y

            current_x = current_pose.position.x
            current_y = current_pose.position.y

            dx = target_x - current_x
            dy = target_y - current_y

            # Calculate distance to target
            distance = math.sqrt(dx*dx + dy*dy)

            # Calculate heading to target
            desired_heading = math.atan2(dy, dx)

            # Get current heading from quaternion
            current_heading = self.get_yaw_from_quaternion(current_pose.orientation)

            # Calculate heading error
            heading_error = desired_heading - current_heading
            # Normalize angle to [-pi, pi]
            while heading_error > math.pi:
                heading_error -= 2 * math.pi
            while heading_error < -math.pi:
                heading_error += 2 * math.pi

            # Simple proportional controller
            cmd_vel.linear.x = min(self.max_linear_speed * 0.5, max(0.1, distance * 0.5))
            cmd_vel.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, heading_error * 2.0))

            # Check for obstacles using laser scan
            if laser_scan and self.is_obstacle_ahead(laser_scan):
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z *= 0.5  # Reduce turning when stopping

        return cmd_vel

    def find_closest_waypoint(self, current_pose, path):
        """Find the closest waypoint on the path"""
        min_dist = float('inf')
        closest_idx = 0

        for i, pose_stamped in enumerate(path):
            dx = pose_stamped.pose.position.x - current_pose.position.x
            dy = pose_stamped.pose.position.y - current_pose.position.y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        return closest_idx

    def find_lookahead_point(self, current_pose, path, start_idx):
        """Find the point on the path that is the lookahead distance away"""
        current_x = current_pose.position.x
        current_y = current_pose.position.y

        for i in range(start_idx, len(path)):
            dx = path[i].pose.position.x - current_x
            dy = path[i].pose.position.y - current_y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist >= self.lookahead_distance:
                return i

        # If no point is far enough, return the last point
        return len(path) - 1 if path else None

    def get_yaw_from_quaternion(self, quat):
        """Extract yaw angle from quaternion"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def is_obstacle_ahead(self, laser_scan):
        """Check if there's an obstacle directly ahead using laser scan"""
        if not laser_scan:
            return False

        # Check the front 30 degrees of the laser scan
        num_ranges = len(laser_scan.ranges)
        front_start = num_ranges // 2 - num_ranges // 24  # -15 degrees
        front_end = num_ranges // 2 + num_ranges // 24    # +15 degrees

        if front_start < 0:
            front_start = 0
        if front_end >= num_ranges:
            front_end = num_ranges - 1

        # Check for obstacles within 1 meter in front
        for i in range(front_start, front_end + 1):
            if not math.isnan(laser_scan.ranges[i]) and laser_scan.ranges[i] < 1.0:
                return True

        return False
```

## 8.5 Motion Planning and Trajectory Generation

### 8.5.1 Trajectory Optimization

Motion planning generates dynamically feasible trajectories:

```python
class TrajectoryOptimizer:
    def __init__(self):
        self.max_velocity = 1.0
        self.max_acceleration = 0.5
        self.dt = 0.1  # Time step

    def generate_trajectory(self, path, start_velocity=(0.0, 0.0)):
        """Generate smooth trajectory following the path"""
        if len(path) < 2:
            return []

        trajectory = []
        current_vel = start_velocity

        for i in range(len(path) - 1):
            start_pos = (path[i].pose.position.x, path[i].pose.position.y)
            end_pos = (path[i+1].pose.position.x, path[i+1].pose.position.y)

            # Calculate distance and direction
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)

            if distance > 0.01:  # Avoid division by zero
                direction = (dx/distance, dy/distance)

                # Calculate required acceleration to reach target velocity
                target_speed = min(self.max_velocity, distance / self.dt)
                required_accel = (target_speed - current_vel[0]) / self.dt

                # Limit acceleration
                if abs(required_accel) > self.max_acceleration:
                    required_accel = math.copysign(self.max_acceleration, required_accel)

                # Update velocity
                new_vel_x = current_vel[0] + required_accel * self.dt
                new_vel_x = max(-self.max_velocity, min(self.max_velocity, new_vel_x))

                # Calculate position update
                new_pos_x = start_pos[0] + new_vel_x * self.dt * direction[0]
                new_pos_y = start_pos[1] + new_vel_x * self.dt * direction[1]

                # Create trajectory point
                traj_point = PoseStamped()
                traj_point.pose.position.x = new_pos_x
                traj_point.pose.position.y = new_pos_y
                traj_point.pose.position.z = 0.0

                # Calculate orientation
                traj_point.pose.orientation.z = math.sin(math.atan2(dy, dx) / 2)
                traj_point.pose.orientation.w = math.cos(math.atan2(dy, dx) / 2)

                trajectory.append(traj_point)
                current_vel = (new_vel_x, 0.0)

        return trajectory

    def smooth_path(self, path, alpha=0.1, beta=0.3, tolerance=0.001):
        """Smooth path using gradient descent optimization"""
        if len(path) < 3:
            return path

        # Convert path to numpy arrays for easier manipulation
        orig_path = np.array([[p.pose.position.x, p.pose.position.y] for p in path])
        smooth_path = orig_path.copy()

        change = tolerance
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(orig_path) - 1):
                for j in range(2):  # x and y coordinates
                    aux = smooth_path[i, j]
                    # Gradient descent step
                    smooth_path[i, j] += alpha * (orig_path[i, j] - smooth_path[i, j])
                    smooth_path[i, j] += beta * (smooth_path[i-1, j] + smooth_path[i+1, j] - 2.0 * smooth_path[i, j])
                    change += abs(aux - smooth_path[i, j])

        # Convert back to PoseStamped format
        smoothed_path = []
        for point in smooth_path:
            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = point[0]
            pose_stamped.pose.position.y = point[1]
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            smoothed_path.append(pose_stamped)

        return smoothed_path
```

### 8.5.2 Dynamic Window Approach (DWA)

DWA is a local path planning method that considers robot dynamics:

```python
class DynamicWindowApproach:
    def __init__(self):
        # Robot parameters
        self.max_speed = 0.5  # m/s
        self.min_speed = 0.0  # m/s
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # rad/s
        self.max_accel = 0.5  # m/ss
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # rad/ss
        self.dt = 0.1  # [s]
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0

    def plan_local_path(self, current_state, goal, obstacles):
        """Plan local path using Dynamic Window Approach"""
        # Generate dynamic window
        u, predicted_trajectory = self.calc_dynamic_window(current_state)

        min_cost = float('inf')
        best_u = [0.0, 0.0]
        best_trajectory = []

        # Evaluate all possible velocities
        for vel in u:
            trajectory = self.predict_trajectory(current_state, vel[0], vel[1])

            # Calculate costs
            to_goal_cost = self.calc_to_goal_cost(trajectory, goal)
            speed_cost = self.calc_speed_cost(trajectory)
            obstacle_cost = self.calc_obstacle_cost(trajectory, obstacles)

            # Total cost
            final_cost = (self.to_goal_cost_gain * to_goal_cost +
                         self.speed_cost_gain * speed_cost +
                         self.obstacle_cost_gain * obstacle_cost)

            # Find minimum cost trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [vel[0], vel[1]]
                best_trajectory = trajectory

        return best_u, best_trajectory

    def calc_dynamic_window(self, current_state):
        """Calculate dynamic window"""
        # Dynamic window from robot specification
        Vs = [self.min_speed, self.max_speed,
              -self.max_yaw_rate, self.max_yaw_rate]

        # Dynamic window from motion model
        Vd = [current_state[3] - self.max_accel * self.dt,
              current_state[3] + self.max_accel * self.dt,
              current_state[4] - self.max_delta_yaw_rate * self.dt,
              current_state[4] + self.max_delta_yaw_rate * self.dt]

        # Find section between Vs and Vd
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw, []

    def predict_trajectory(self, current_state, v, y):
        """Predict trajectory with given velocity"""
        x = current_state[:]
        trajectory = [x[:]]

        time = 0
        while time <= self.predict_time:
            x = self.motion(x, [v, y])
            trajectory.append(x[:])
            time += self.dt

        return trajectory

    def motion(self, state, vel):
        """Motion model"""
        state[0] += vel[0] * math.cos(state[2]) * self.dt
        state[1] += vel[0] * math.sin(state[2]) * self.dt
        state[2] += vel[1] * self.dt
        state[3] = vel[0]
        state[4] = vel[1]

        return state

    def calc_to_goal_cost(self, trajectory, goal):
        """Calculate cost to goal"""
        dx = goal[0] - trajectory[-1][0]
        dy = goal[1] - trajectory[-1][1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1][2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

        return cost

    def calc_speed_cost(self, trajectory):
        """Calculate speed cost"""
        # Higher speeds are preferred
        return abs(trajectory[-1][3])  # Velocity at the end of trajectory

    def calc_obstacle_cost(self, trajectory, obstacles):
        """Calculate obstacle cost"""
        min_dist = float('inf')
        for i in range(len(trajectory)):
            for j in range(len(obstacles)):
                dist = math.sqrt((trajectory[i][0] - obstacles[j][0])**2 +
                                (trajectory[i][1] - obstacles[j][1])**2)
                if dist <= min_dist:
                    min_dist = dist

        # If min_dist is infinity, it means no obstacle is in the trajectory
        if min_dist == float('inf'):
            return 0

        # Cost is inversely proportional to distance
        return 1.0 / min_dist if min_dist != 0 else float('inf')
```

## 8.6 Behavior-Based Navigation

### 8.6.1 Reactive Navigation Behaviors

Behavior-based navigation decomposes navigation into simple, reactive behaviors:

```python
class BehaviorBasedNavigator:
    def __init__(self):
        self.behaviors = [
            self.go_to_goal_behavior,
            self.avoid_obstacles_behavior,
            self.follow_walls_behavior
        ]
        self.current_behavior = 0

    def go_to_goal_behavior(self, current_pose, goal_pose, sensor_data):
        """Simple go-to-goal behavior"""
        dx = goal_pose.position.x - current_pose.position.x
        dy = goal_pose.position.y - current_pose.position.y

        distance = math.sqrt(dx*dx + dy*dy)
        angle_to_goal = math.atan2(dy, dx)

        current_yaw = self.get_yaw_from_quaternion(current_pose.orientation)
        angle_error = angle_to_goal - current_yaw

        # Normalize angle
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        # Return velocity command
        linear_vel = min(0.5, distance * 0.5) if distance > 0.2 else 0.0
        angular_vel = max(-1.0, min(1.0, angle_error * 2.0))

        return linear_vel, angular_vel

    def avoid_obstacles_behavior(self, current_pose, goal_pose, sensor_data):
        """Obstacle avoidance behavior using sensor data"""
        if not sensor_data or len(sensor_data.ranges) == 0:
            return 0.0, 0.0

        # Check for obstacles in front
        num_ranges = len(sensor_data.ranges)
        front_start = num_ranges // 2 - num_ranges // 8  # -22.5 degrees
        front_end = num_ranges // 2 + num_ranges // 8    # +22.5 degrees

        min_range = min(sensor_data.ranges[front_start:front_end]) if front_start < front_end else float('inf')

        if min_range < 0.5:  # Obstacle too close
            # Turn away from obstacle
            left_clear = min(sensor_data.ranges[num_ranges//2:num_ranges*3//4]) > 1.0
            right_clear = min(sensor_data.ranges[num_ranges//4:num_ranges//2]) > 1.0

            if left_clear and not right_clear:
                return 0.0, 0.5  # Turn left
            elif right_clear and not left_clear:
                return 0.0, -0.5  # Turn right
            else:
                return 0.0, 0.5  # Default to turning left

        # If no obstacle ahead, move forward
        return 0.3, 0.0

    def follow_walls_behavior(self, current_pose, goal_pose, sensor_data):
        """Wall following behavior"""
        if not sensor_data or len(sensor_data.ranges) == 0:
            return 0.0, 0.0

        # Use sensor values from different directions
        left_range = sensor_data.ranges[len(sensor_data.ranges) * 3 // 4]  # 135 degrees
        front_range = sensor_data.ranges[len(sensor_data.ranges) // 2]     # 0 degrees
        right_range = sensor_data.ranges[len(sensor_data.ranges) // 4]     # -45 degrees

        # Simple wall following algorithm
        target_distance = 0.5  # Desired distance from wall

        if front_range > 1.0:  # Path is clear ahead
            if left_range > target_distance * 1.5:  # Too far from left wall
                return 0.3, -0.3  # Turn left toward wall
            elif left_range < target_distance * 0.7:  # Too close to left wall
                return 0.3, 0.3   # Turn right away from wall
            else:  # Good distance from wall
                return 0.3, 0.0   # Move forward
        else:  # Obstacle ahead
            return 0.0, 0.5       # Turn away from obstacle

    def get_yaw_from_quaternion(self, quat):
        """Extract yaw from quaternion"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def select_behavior(self, current_pose, goal_pose, sensor_data):
        """Select the most appropriate behavior based on the situation"""
        # This is a simple priority-based selection
        # In practice, you might use a more sophisticated arbitration mechanism

        # Check for immediate obstacles
        if sensor_data and len(sensor_data.ranges) > 0:
            front_ranges = sensor_data.ranges[len(sensor_data.ranges)//2 - len(sensor_data.ranges)//16 :
                                            len(sensor_data.ranges)//2 + len(sensor_data.ranges)//16]
            min_front_range = min(front_ranges) if front_ranges else float('inf')

            if min_front_range < 0.4:  # Emergency obstacle avoidance
                return self.avoid_obstacles_behavior(current_pose, goal_pose, sensor_data)

        # For normal navigation, use go-to-goal behavior
        return self.go_to_goal_behavior(current_pose, goal_pose, sensor_data)
```

## 8.7 Integration with Physical AI Systems

Navigation systems in Physical AI must consider the interaction between perception, planning, and control:

### 8.7.1 Perception-Action Integration

```python
class IntegratedNavigationSystem(Node):
    def __init__(self):
        super().__init__('integrated_navigation_system')

        # Initialize components
        self.perception_module = PerceptionModule()
        self.mapping_module = OccupancyGridMap(100, 100, 0.1)
        self.path_planner = AStarPlanner(None)
        self.local_planner = LocalPlanner()
        self.behavior_navigator = BehaviorBasedNavigator()

        # Setup ROS 2 interface
        self.setup_ros_interface()

    def setup_ros_interface(self):
        """Setup ROS 2 publishers and subscribers"""
        # Subscribe to sensor data
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10
        )

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        # Setup main navigation timer
        self.nav_timer = self.create_timer(0.1, self.integrated_navigation_loop)

    def integrated_navigation_loop(self):
        """Main loop integrating perception, mapping, and navigation"""
        # Update map with latest sensor data
        if self.latest_scan and self.current_odom:
            robot_pose = self.odom_to_pose(self.current_odom)
            self.mapping_module.update_with_laser_scan(
                (robot_pose.position.x, robot_pose.position.y, self.get_yaw_from_quaternion(robot_pose.orientation)),
                self.latest_scan.ranges,
                self.generate_scan_angles(self.latest_scan)
            )

            # Update path planner with new map
            self.path_planner.grid_map = self.mapping_module.grid

            # Perform navigation
            if self.navigation_goal:
                self.execute_navigation(robot_pose)

    def execute_navigation(self, robot_pose):
        """Execute navigation with integrated perception and planning"""
        # Use behavior-based navigation as fallback
        linear_vel, angular_vel = self.behavior_navigator.select_behavior(
            robot_pose, self.navigation_goal, self.latest_scan
        )

        # Create and publish velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_vel
        cmd_vel.angular.z = angular_vel
        self.cmd_vel_pub.publish(cmd_vel)

    def laser_callback(self, msg):
        """Handle laser scan data"""
        self.latest_scan = msg

    def odom_callback(self, msg):
        """Handle odometry data"""
        self.current_odom = msg

    def odom_to_pose(self, odom_msg):
        """Convert odometry message to pose"""
        pose_stamped = PoseStamped()
        pose_stamped.pose = odom_msg.pose.pose
        return pose_stamped.pose
```

## 8.8 Performance Evaluation and Safety

### 8.8.1 Navigation Metrics

Evaluating navigation system performance:

```python
class NavigationMetrics:
    def __init__(self):
        self.path_length = 0.0
        self.execution_time = 0.0
        self.success_count = 0
        self.failure_count = 0
        self.collision_count = 0
        self.oscillation_count = 0

    def calculate_path_efficiency(self, planned_path, optimal_path_length):
        """Calculate path efficiency ratio"""
        if optimal_path_length > 0:
            return self.path_length / optimal_path_length
        return 1.0  # If no optimal path, consider it 100% efficient

    def calculate_success_rate(self):
        """Calculate navigation success rate"""
        total_attempts = self.success_count + self.failure_count
        if total_attempts > 0:
            return self.success_count / total_attempts
        return 0.0

    def evaluate_safety(self):
        """Evaluate navigation safety based on collisions and oscillations"""
        safety_score = 1.0
        if self.collision_count > 0:
            safety_score -= 0.5 * self.collision_count  # Heavy penalty for collisions
        if self.oscillation_count > 0:
            safety_score -= 0.1 * self.oscillation_count  # Lighter penalty for oscillations

        return max(0.0, safety_score)  # Ensure non-negative score
```

## Summary

Mobile robot navigation and path planning form the foundation of autonomous robotic systems. The integration of localization, mapping, path planning, and motion control enables robots to operate effectively in complex environments. Modern navigation systems leverage both classical algorithms like A* and Dijkstra, as well as sampling-based methods like RRT for high-dimensional spaces. The ROS 2 Navigation2 stack provides a comprehensive framework for implementing these concepts. As Physical AI systems become more sophisticated, navigation algorithms must continue to evolve to handle dynamic environments, uncertainty, and safety-critical applications.