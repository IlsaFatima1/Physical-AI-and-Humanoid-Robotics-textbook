# Chapter 8 Quiz: Mobile Robot Navigation and Path Planning

## Multiple Choice Questions

1. What does SLAM stand for in robotics navigation?
   a) Simultaneous Localization and Mapping
   b) Systematic Localization and Mapping
   c) Simultaneous Learning and Mapping
   d) Systematic Learning and Autonomous Mapping

2. Which path planning algorithm is guaranteed to find the shortest path?
   a) Dijkstra's algorithm
   b) Greedy best-first search
   c) Rapidly-exploring Random Trees (RRT)
   d) Potential fields

3. What is the main advantage of the A* algorithm over Dijkstra's algorithm?
   a) A* is faster due to heuristic guidance
   b) A* finds paths in 3D space while Dijkstra only works in 2D
   c) A* can handle dynamic obstacles
   d) A* requires less memory

4. In the Dynamic Window Approach (DWA), what does the "dynamic window" represent?
   a) A window in the robot's GUI
   b) The set of all possible velocities given robot dynamics
   c) A visual window for obstacle detection
   d) The robot's field of view

5. What is the primary purpose of a costmap in ROS Navigation?
   a) To store the robot's trajectory
   b) To represent obstacles and navigation costs in the environment
   c) To store sensor calibration data
   d) To maintain communication logs

## Practical Application Questions

6. You are developing a navigation system for a mobile robot in a dynamic warehouse environment. Design a complete navigation pipeline that includes:
   a) Localization approach suitable for the environment
   b) Path planning strategy for static obstacles
   c) Local planning approach for dynamic obstacle avoidance
   d) Safety mechanisms to prevent collisions with moving objects

7. Implement a hybrid navigation system that combines global path planning with local obstacle avoidance. Describe:
   a) How the global and local planners interact
   b) The conditions that trigger replanning
   c) How to handle situations where local planner cannot find a way around obstacles
   d) Methods to ensure smooth transitions between global path following and local避ing

8. Design a navigation system for a robot that must operate in GPS-denied environments. Consider:
   a) Appropriate localization methods
   b) Mapping strategies for large-scale environments
   c) Path planning approaches that handle uncertainty
   d) Recovery strategies when navigation fails

## Code Analysis Questions

9. Analyze the following A* path planning code and identify potential issues:
   ```python
   def a_star(grid, start, goal):
       open_set = [start]
       came_from = {}
       g_score = {start: 0}
       f_score = {start: heuristic(start, goal)}

       while open_set:
           current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

           if current == goal:
               return reconstruct_path(came_from, current)

           open_set.remove(current)

           for neighbor in get_neighbors(current):
               if neighbor in [0, 0, 0]:  # Check if neighbor is valid
                   continue

               tentative_g_score = g_score[current] + distance(current, neighbor)

               if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                   came_from[neighbor] = current
                   g_score[neighbor] = tentative_g_score
                   f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                   if neighbor not in open_set:
                       open_set.append(neighbor)

       return []  # No path found
   ```

10. The following navigation control code has potential problems. Identify and correct them:
    ```python
    def navigate_to_goal(robot_pose, goal_pose, obstacles):
        # Calculate angle to goal
        dx = goal_pose.x - robot_pose.x
        dy = goal_pose.y - robot_pose.y
        angle_to_goal = math.atan2(dx, dy)  # Wrong order of arguments

        # Get robot's current orientation
        current_angle = robot_pose.theta

        # Calculate error
        angle_error = angle_to_goal - current_angle

        # Set velocities
        linear_vel = 1.0  # Constant speed regardless of distance
        angular_vel = angle_error * 2.0

        # Check for obstacles
        for obs in obstacles:
            dist_to_obs = math.sqrt((robot_pose.x - obs.x)**2 + (robot_pose.y - obs.y)**2)
            if dist_to_obs < 1.0:  # Obstacle too close
                linear_vel = 0.0
                angular_vel = 1.0  # Always turn right regardless of obstacle position

        return linear_vel, angular_vel
    ```

## Conceptual Questions

11. Explain the differences between global path planning and local path planning. When would you use each approach, and how do they complement each other in a complete navigation system?

12. Describe the main challenges of navigation in dynamic environments and propose solutions for handling moving obstacles, changing maps, and uncertain localization.

13. Compare and contrast different path planning algorithms (A*, Dijkstra, RRT, Potential Fields). What are the advantages and disadvantages of each, and when would you choose one over the others?

14. How does uncertainty affect robot navigation, and what techniques can be used to handle uncertainty in perception, localization, and mapping? How does this impact path planning decisions?

---

## Answer Key

### Multiple Choice Answers:
1. a) Simultaneous Localization and Mapping
2. a) Dijkstra's algorithm
3. a) A* is faster due to heuristic guidance
4. b) The set of all possible velocities given robot dynamics
5. b) To represent obstacles and navigation costs in the environment

### Practical Application Answers:

6. Warehouse navigation pipeline:
   a) Localization: AMCL (Adaptive Monte Carlo Localization) with good landmarks
   b) Path planning: Global A* on static costmap with inflation for safety
   c) Local planning: DWA or TEB (Timed Elastic Band) for dynamic obstacle avoidance
   d) Safety: Emergency stops, safety corridors, velocity limiting near humans

7. Hybrid navigation system:
   a) Global planner provides path, local planner follows while avoiding obstacles
   b) Replan when path is blocked, goal changes, or robot deviates significantly
   c) Use recovery behaviors like clearing costmap or using backup plans
   d) Implement smooth velocity profiles and path smoothing

8. GPS-denied navigation:
   a) Localization: Visual-inertial odometry, LiDAR SLAM, or UWB positioning
   b) Mapping: Occupancy grid maps with loop closure detection
   c) Planning: Probabilistic roadmaps or sampling-based methods that consider uncertainty
   d) Recovery: Safe stop, return to known location, or human intervention

### Code Analysis Answers:

9. Issues with the A* code:
   - Neighbor validation is incorrect (`if neighbor in [0, 0, 0]` compares to a list)
   - No check if neighbor is in grid bounds
   - No check if neighbor is occupied
   - Should be: `if not is_valid_neighbor(neighbor, grid): continue`
   - Missing proper neighbor validation function

10. Issues with navigation control:
   - Wrong order of arguments in atan2 (should be atan2(dy, dx))
   - Constant linear velocity doesn't account for distance to goal
   - Always turns right regardless of obstacle position
   - No proper obstacle avoidance direction calculation
   - Corrections:
   ```python
   angle_to_goal = math.atan2(dy, dx)  # Correct order
   linear_vel = min(max_speed, distance_to_goal * gain)  # Speed based on distance

   # For obstacle avoidance, calculate direction away from obstacles
   if dist_to_obs < 1.0:
       linear_vel = 0.0
       # Calculate turn direction based on obstacle position
       obs_angle = math.atan2(obs.y - robot_pose.y, obs.x - robot_pose.x)
       turn_direction = math.atan2(dy, dx) - obs_angle
       angular_vel = math.copysign(1.0, turn_direction)
   ```

### Conceptual Answers:

11. Global vs. Local planning:
   - Global planning: Computes path from start to goal using static map, computationally intensive but optimal
   - Local planning: Short-term path following with obstacle avoidance using sensor data, reactive and fast
   - Used together: Global provides path, local executes while handling dynamic obstacles
   - Complement: Global for long-term strategy, local for short-term tactics

12. Dynamic environment challenges:
   - Moving obstacles: Predict motion, use velocity obstacles, temporal planning
   - Changing maps: Continuous mapping, dynamic object tracking, predictive models
   - Uncertain localization: Sensor fusion, particle filters, map matching
   - Solutions: Temporal planning, predictive models, robust localization

13. Path planning algorithm comparison:
   - A*: Optimal, fast with good heuristic, complete
   - Dijkstra: Optimal, no heuristic needed, slower than A*
   - RRT: Good for high-dimensional spaces, probabilistically complete, not optimal
   - Potential Fields: Simple, reactive, susceptible to local minima
   - Choose based on environment complexity, dimensionality, and optimality requirements

14. Uncertainty handling:
   - Perception: Sensor fusion, uncertainty modeling, robust feature detection
   - Localization: Particle filters, Kalman filters, multiple hypothesis tracking
   - Mapping: Probabilistic maps, occupancy grids with uncertainty
   - Path planning: Risk-aware planning, probabilistic roadmaps, chance-constrained optimization
   - Impact: More conservative paths, safety margins, replanning when uncertainty is high