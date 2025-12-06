"""
Mobile Robot Navigation and Path Planning Examples
Chapter 8: Mobile Robot Navigation and Path Planning

This module demonstrates various navigation and path planning concepts including:
- SLAM implementation
- Path planning algorithms (A*, Dijkstra, RRT)
- Local navigation and obstacle avoidance
- ROS 2 navigation stack integration
- Trajectory optimization
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformListener, Buffer
import tf2_ros
import numpy as np
import math
import heapq
from typing import List, Tuple
import random


class NavigationSystemNode(Node):
    """
    Comprehensive navigation system integrating SLAM, path planning, and local navigation.
    """
    def __init__(self):
        super().__init__('navigation_system')

        # Navigation state
        self.current_pose = None
        self.current_odom = None
        self.navigation_goal = None
        self.map = None
        self.global_path = None
        self.laser_scan = None

        # Navigation components
        self.occupancy_grid_map = None
        self.global_planner = None
        self.local_planner = LocalPlanner()
        self.behavior_navigator = BehaviorBasedNavigator()

        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Setup ROS 2 interface
        self.setup_ros_interface()

        self.get_logger().info("Navigation System initialized")

    def setup_ros_interface(self):
        """Setup ROS 2 publishers and subscribers"""
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/plan', 10)
        self.global_costmap_pub = self.create_publisher(OccupancyGrid, '/global_costmap', 10)

        # Setup navigation timer
        self.nav_timer = self.create_timer(0.1, self.navigation_loop)

    def odom_callback(self, msg):
        """Update robot odometry"""
        self.current_odom = msg
        # Convert odometry to pose
        self.current_pose = msg.pose.pose

    def scan_callback(self, msg):
        """Update laser scan data"""
        self.laser_scan = msg

        # Update occupancy grid map if we have a map
        if self.occupancy_grid_map and self.current_pose:
            robot_x = self.current_pose.position.x
            robot_y = self.current_pose.position.y
            # Get robot's orientation from quaternion
            quat = self.current_pose.orientation
            siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
            cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
            robot_theta = math.atan2(siny_cosp, cosy_cosp)

            # Update map with scan data
            self.occupancy_grid_map.update_with_laser_scan(
                (robot_x, robot_y, robot_theta),
                list(msg.ranges),
                self.generate_scan_angles(msg)
            )

    def generate_scan_angles(self, scan_msg):
        """Generate angles for each range measurement in laser scan"""
        angles = []
        angle = scan_msg.angle_min
        for _ in range(len(scan_msg.ranges)):
            angles.append(angle)
            angle += scan_msg.angle_increment
        return angles

    def map_callback(self, msg):
        """Update map data"""
        self.map = msg
        # Initialize occupancy grid map if not already done
        if self.occupancy_grid_map is None:
            self.occupancy_grid_map = OccupancyGridMap(
                msg.info.width, msg.info.height, msg.info.resolution
            )
            # Set map origin
            self.occupancy_grid_map.origin_x = msg.info.origin.position.x
            self.occupancy_grid_map.origin_y = msg.info.origin.position.y

    def goal_callback(self, msg):
        """Set navigation goal"""
        self.navigation_goal = msg.pose
        self.get_logger().info(f"New goal set: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

        # Plan path to goal if we have current pose and map
        if self.current_pose and self.map:
            self.plan_global_path()

    def plan_global_path(self):
        """Plan global path from current position to goal"""
        if not self.current_pose or not self.navigation_goal or self.map is None:
            return

        # Convert poses to grid coordinates
        start_grid = self.world_to_grid(
            self.current_pose.position.x,
            self.current_pose.position.y
        )
        goal_grid = self.world_to_grid(
            self.navigation_goal.position.x,
            self.navigation_goal.position.y
        )

        # Create path planner
        if self.occupancy_grid_map:
            self.global_planner = AStarPlanner(self.occupancy_grid_map.grid)

        # Plan path
        path_grid = self.global_planner.plan_path(start_grid, goal_grid) if self.global_planner else []

        if path_grid:
            # Convert grid path back to world coordinates
            self.global_path = []
            for grid_x, grid_y in path_grid:
                world_x, world_y = self.grid_to_world(grid_x, grid_y)
                pose_stamped = PoseStamped()
                pose_stamped.pose.position.x = world_x
                pose_stamped.pose.position.y = world_y
                pose_stamped.pose.position.z = 0.0
                pose_stamped.pose.orientation.w = 1.0  # No rotation
                self.global_path.append(pose_stamped)

            # Publish path
            path_msg = Path()
            path_msg.poses = self.global_path
            path_msg.header.frame_id = "map"
            path_msg.header.stamp = self.get_clock().now().to_msg()
            self.path_pub.publish(path_msg)

            self.get_logger().info(f"Global path planned with {len(self.global_path)} waypoints")
        else:
            self.get_logger().warn("No global path found to goal")

    def navigation_loop(self):
        """Main navigation control loop"""
        if self.current_pose and self.navigation_goal:
            # Use behavior-based navigation as primary approach
            linear_vel, angular_vel = self.behavior_navigator.select_behavior(
                self.current_pose, self.navigation_goal, self.laser_scan
            )

            # Create and publish velocity command
            cmd_vel = Twist()
            cmd_vel.linear.x = linear_vel
            cmd_vel.angular.z = angular_vel
            self.cmd_vel_pub.publish(cmd_vel)

            # Publish updated costmap
            if self.occupancy_grid_map:
                costmap_msg = self.occupancy_grid_map.to_occupancy_grid_msg()
                costmap_msg.header.frame_id = "map"
                costmap_msg.header.stamp = self.get_clock().now().to_msg()
                self.global_costmap_pub.publish(costmap_msg)

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


class OccupancyGridMap:
    """
    Occupancy grid mapping implementation for SLAM and navigation.
    """
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

        # Sensor parameters for mapping
        self.free_threshold = 0.3
        self.occupied_threshold = 0.6
        self.max_range = 30.0  # Maximum sensor range

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

        # Convert scan to world coordinates and update grid
        for i, (range_val, angle) in enumerate(zip(scan_ranges, scan_angles)):
            if range_val < 0.1 or range_val > self.max_range:  # Invalid range
                continue

            # Calculate endpoint of this ray in world coordinates
            world_x = robot_x + range_val * math.cos(robot_theta + angle)
            world_y = robot_y + range_val * math.sin(robot_theta + angle)

            # Get grid coordinates
            end_grid_x, end_grid_y = self.world_to_grid(world_x, world_y)
            start_grid_x, start_grid_y = self.world_to_grid(robot_x, robot_y)

            # Perform ray tracing to update grid
            self.ray_trace(start_grid_x, start_grid_y, end_grid_x, end_grid_y, range_val < self.max_range * 0.9)

    def ray_trace(self, x0, y0, x1, y1, endpoint_occupied):
        """Perform ray tracing between two points using Bresenham's algorithm"""
        # Check bounds
        if (not (0 <= x0 < self.width and 0 <= y0 < self.height) or
            not (0 <= x1 < self.width and 0 <= y1 < self.height)):
            return

        # Bresenham's line algorithm with probability updates
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x_step = 1 if x0 < x1 else -1
        y_step = 1 if y0 < y1 else -1

        error = dx - dy
        x, y = x0, y0

        # Update cells along the ray (free space)
        while x != x1 or y != y1:
            if 0 <= x < self.width and 0 <= y < self.height:
                # Update log odds for free space
                self.log_odds[y, x] += self.log_odds_from_probability(0.4)

            if error * 2 > -dy:
                error -= dy
                x += x_step
            if error * 2 < dx:
                error += dx
                y += y_step

        # Update endpoint
        if 0 <= x1 < self.width and 0 <= y1 < self.height:
            if endpoint_occupied:
                # Update log odds for occupied space
                self.log_odds[y1, x1] += self.log_odds_from_probability(0.8)
            else:
                # Update log odds for free space
                self.log_odds[y1, x1] += self.log_odds_from_probability(0.3)

    def log_odds_from_probability(self, prob):
        """Convert probability to log odds"""
        prob = max(0.01, min(0.99, prob))  # Clamp to avoid log(0)
        return math.log(prob / (1 - prob))

    def probability_from_log_odds(self, log_odds):
        """Convert log odds to probability"""
        prob = 1 - 1 / (1 + math.exp(log_odds))
        return min(1.0, max(0.0, prob))

    def get_probability(self, x, y):
        """Get occupancy probability at world coordinates"""
        grid_x, grid_y = self.world_to_grid(x, y)
        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            return self.probability_from_log_odds(self.log_odds[grid_y, grid_x])
        return 0.5  # Unknown area

    def is_free(self, x, y):
        """Check if a location is free (not occupied)"""
        prob = self.get_probability(x, y)
        return prob < self.occupied_threshold

    def to_occupancy_grid_msg(self):
        """Convert to ROS OccupancyGrid message"""
        from nav_msgs.msg import OccupancyGrid
        from std_msgs.msg import Header

        grid_msg = OccupancyGrid()
        grid_msg.header = Header()
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.width
        grid_msg.info.height = self.height
        grid_msg.info.origin.position.x = self.origin_x
        grid_msg.info.origin.position.y = self.origin_y
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

        # Convert probabilities to int8 values (-1: unknown, 0-100: probability)
        data = []
        for y in range(self.height):
            for x in range(self.width):
                prob = self.probability_from_log_odds(self.log_odds[y, x])
                if prob < 0.25:
                    val = 0  # Free space
                elif prob > 0.75:
                    val = 100  # Occupied space
                else:
                    val = -1  # Unknown
                data.append(val)

        grid_msg.data = data
        return grid_msg


class AStarPlanner:
    """
    A* path planning algorithm implementation.
    """
    def __init__(self, grid_map):
        self.grid_map = grid_map
        self.height, self.width = grid_map.shape if grid_map is not None else (0, 0)

    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Plan path using A* algorithm"""
        if (not self.is_valid_cell(start[0], start[1]) or
            not self.is_valid_cell(goal[0], goal[1])):
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
        """Get valid neighbors of a cell (8-connected)"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip current cell
                nx, ny = x + dx, y + dy
                if self.is_valid_cell(nx, ny):
                    neighbors.append((nx, ny))
        return neighbors

    def is_valid_cell(self, x: int, y: int) -> bool:
        """Check if cell is valid and not occupied"""
        if (self.grid_map is None or
            x < 0 or x >= self.width or y < 0 or y >= self.height):
            return False
        # Consider cell as free if occupancy probability is below threshold
        return self.grid_map[y, x] < 0.7

    def reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


class LocalPlanner:
    """
    Local navigation planner for obstacle avoidance and path following.
    """
    def __init__(self):
        self.lookahead_distance = 1.0
        self.max_linear_speed = 0.5
        self.max_angular_speed = 1.0
        self.min_obstacle_distance = 0.5

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
            cmd_vel.linear.x = min(self.max_linear_speed, max(0.1, distance * 0.5))
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
        if not laser_scan or len(laser_scan.ranges) == 0:
            return False

        # Check the front 30 degrees of the laser scan
        num_ranges = len(laser_scan.ranges)
        front_start = num_ranges // 2 - num_ranges // 24  # -15 degrees
        front_end = num_ranges // 2 + num_ranges // 24    # +15 degrees

        if front_start < 0:
            front_start = 0
        if front_end >= num_ranges:
            front_end = num_ranges - 1

        # Check for obstacles within minimum distance in front
        for i in range(front_start, front_end + 1):
            if (not math.isnan(laser_scan.ranges[i]) and
                laser_scan.ranges[i] < self.min_obstacle_distance):
                return True

        return False


class BehaviorBasedNavigator:
    """
    Behavior-based navigation system with multiple reactive behaviors.
    """
    def __init__(self):
        self.behaviors = [
            self.go_to_goal_behavior,
            self.avoid_obstacles_behavior,
            self.wall_following_behavior
        ]
        self.current_behavior = 0
        self.safety_distance = 0.5

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

        if front_start < 0:
            front_start = 0
        if front_end >= num_ranges:
            front_end = num_ranges - 1

        min_range = min(sensor_data.ranges[front_start:front_end+1]) if front_start <= front_end else float('inf')

        if min_range < self.safety_distance:  # Obstacle too close
            # Turn away from obstacle
            left_clear = min(sensor_data.ranges[num_ranges//2:num_ranges*3//4]) > 1.0 if num_ranges*3//4 <= num_ranges else True
            right_clear = min(sensor_data.ranges[num_ranges//4:num_ranges//2+1]) > 1.0 if num_ranges//2+1 <= num_ranges else True

            if left_clear and not right_clear:
                return 0.0, 0.5  # Turn left
            elif right_clear and not left_clear:
                return 0.0, -0.5  # Turn right
            else:
                return 0.0, 0.5  # Default to turning left

        # If no obstacle ahead, move forward
        return 0.3, 0.0

    def wall_following_behavior(self, current_pose, goal_pose, sensor_data):
        """Wall following behavior"""
        if not sensor_data or len(sensor_data.ranges) == 0:
            return 0.0, 0.0

        # Use sensor values from different directions
        left_idx = int(len(sensor_data.ranges) * 3 / 4)  # 135 degrees
        front_idx = len(sensor_data.ranges) // 2         # 0 degrees
        right_idx = int(len(sensor_data.ranges) / 4)     # -45 degrees

        left_range = sensor_data.ranges[left_idx] if left_idx < len(sensor_data.ranges) else float('inf')
        front_range = sensor_data.ranges[front_idx] if front_idx < len(sensor_data.ranges) else float('inf')
        right_range = sensor_data.ranges[right_idx] if right_idx < len(sensor_data.ranges) else float('inf')

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
        # Emergency obstacle avoidance has highest priority
        if sensor_data and len(sensor_data.ranges) > 0:
            front_ranges = sensor_data.ranges[len(sensor_data.ranges)//2 - len(sensor_data.ranges)//16 :
                                            len(sensor_data.ranges)//2 + len(sensor_data.ranges)//16]
            min_front_range = min(front_ranges) if front_ranges else float('inf')

            if min_front_range < 0.4:  # Emergency obstacle avoidance
                return self.avoid_obstacles_behavior(current_pose, goal_pose, sensor_data)

        # For normal navigation, use go-to-goal behavior
        return self.go_to_goal_behavior(current_pose, goal_pose, sensor_data)


def main(args=None):
    """Main function to run the navigation system"""
    rclpy.init(args=args)

    # Create navigation system node
    nav_node = NavigationSystemNode()

    try:
        # Spin the node
        rclpy.spin(nav_node)
    except KeyboardInterrupt:
        nav_node.get_logger().info('Navigation system interrupted')
    finally:
        nav_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()