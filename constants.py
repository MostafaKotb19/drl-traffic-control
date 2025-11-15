# Screen Dimensions
SCREEN_WIDTH = 850
SCREEN_HEIGHT = 850

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (100, 100, 100)
DARK_GREY = (50, 50, 50)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0) # Road markings
LIGHT_GREEN = (170, 255, 170) # Background color

# Road and Lane Dimensions
ROAD_WIDTH = 120  # Total width of a 2-lane road
LANE_WIDTH = ROAD_WIDTH / 2
INTERSECTION_SIZE = ROAD_WIDTH # Size of the central intersection box
LANE_DIVIDER_WIDTH = 5
ROAD_DIVIDER_WIDTH = 10

# Vehicle Properties
CAR_WIDTH = 25
CAR_LENGTH = 50
MAX_SPEED = 3  # pixels per frame
MAX_SPEED_VARIATION = 0.5 # +/- variation in max_speed for realism
ACCELERATION = 0.05
BRAKING_DECELERATION = 0.2
SAFE_DISTANCE = 25 # Increased slightly for better spacing

# Simulation Parameters
FPS = 60 # Frames per second
TIMESTEP = 1.0 / FPS # Simulation time per frame
VEHICLE_SPAWN_RATE = 60 

# Traffic Light Properties
LIGHT_RADIUS = 15
GREEN_DURATION = 600 # The duration for the baseline Timer Agent

# --- SCENARIO AND METRIC CONSTANTS ---
SPAWN_RATE_NORMAL = 360
SPAWN_RATE_RUSH_HOUR = 240 # Lower number means more cars
SPAWN_RATE_UNBALANCED_BUSY = 240    # Very high traffic for one direction
SPAWN_RATE_UNBALANCED_QUIET = 540  # Very low traffic for other directions

# --- RL ENVIRONMENT CONSTANTS ---
MAX_STEPS_PER_EPISODE = 1800 # Corresponds to 30 seconds if 1 step = 1 simulation second

# Reward/Penalty values
REWARD_COLLISION_PENALTY = -5000   # Make this penalty EXTREMELY high
DANGER_ZONE_PENALTY = -100           # Penalty for unsafe switching
REWARD_ACTION_PENALTY = -1          # Penalty for switching
REWARD_THROUGHPUT_BONUS = 300        # Bonus for each car that successfully passes

WAIT_PENALTY_EXPONENT = 2          # Use > 1 for non-linear penalty. 2 = quadratic.
WAIT_PENALTY_SCALING_FACTOR = 30   # Scales wait time (in frames). 60 frames = 1 second.