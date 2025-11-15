import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from simulation import Simulation
from gui import GUI
from constants import *

class TrafficEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode: str = None, spawn_rate: int = None, spawn_configs: dict = None):
        """
        Traffic Environment for RL-based Traffic Light Control.
        
        Parameters:
            render_mode: 'human' to enable GUI rendering, None for headless mode
            spawn_rate: Uniform spawn rate for all directions (overridden by spawn_configs if provided)
            spawn_configs: Dictionary specifying spawn rates per direction, e.g., {'N': 300, 'S': 400, 'E': 500, 'W': 600}
        """
        super(TrafficEnv, self).__init__()
                
        if spawn_configs:
            self.spawn_configs = spawn_configs
        elif spawn_rate:
             self.spawn_configs = {d: spawn_rate for d in ['N', 'S', 'E', 'W']}
        else:
             self.spawn_configs = {d: SPAWN_RATE_NORMAL for d in ['N', 'S', 'E', 'W']}
             
        self.sim = Simulation(spawn_configs=self.spawn_configs)
        
        # Action Space: {0: Keep, 1: Switch}
        self.action_space = spaces.Discrete(2)
        
        # Observation Space using Time-To-Intersection (TTI)
        # 0-7: TTI features (N, S, E, W)
        # 8-11: queue_length (N, S, E, W)
        # 12:  current_phase (0 for NS, 1 for EW)
        # 13:  time_in_current_phase
        # 14-17: is_car_in_intersection (N, S, E, W)
        self.observation_space = spaces.Box(
            low=np.zeros(18, dtype=np.float32),
            high=np.array([
                60, 60,      # Max TTI in seconds for N
                60, 60,      # Max TTI in seconds for S
                60, 60,      # Max TTI in seconds for E
                60, 60,      # Max TTI in seconds for W
                50, 50, 50, 50, # Max queue lengths
                1,              # Max phase
                MAX_STEPS_PER_EPISODE,
                1, 1, 1, 1      # Boolean flags for intersection occupancy (NEW)
            ], dtype=np.float32),
            dtype=np.float32
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.gui = None
        if self.render_mode == "human":
            self.gui = GUI(self.sim)

        self.collision_count = 0
        self.time_in_current_phase = 0

    def _get_intersection_occupancy_features(self) -> dict:
        """ 
        Checks if any vehicle for a given direction is currently inside the intersection.
        
        Returns:
            dict: {'N': 0 or 1, 'S': 0 or 1, 'E': 0 or 1, 'W': 0 or 1}
        """
        occupancy = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'W': 0.0}
        for vehicle in self.sim.get_all_vehicles():
            if self.sim.intersection_rect.colliderect(vehicle.rect):
                occupancy[vehicle.direction] = 1.0
        return occupancy
    
    def _get_tti_features(self) -> dict:
        """
        Calculates the Time-To-Intersection (in seconds) for the 2 closest
        approaching vehicles in each direction.

        Returns:
            dict: {'N': {'tti_1': float, 'tti_2': float}, 'S': {...}, 'E': {...}, 'W': {...}}
        """
        # Default to a high TTI value (e.g., 60 seconds) if no car is approaching
        sentinel_value = 60.0 
        features = {}

        for direction in ['N', 'S', 'E', 'W']:
            ttis = []
            for vehicle in self.sim.get_all_vehicles():
                dist = self.sim._get_dist_to_intersection(vehicle)
                # Consider only vehicles of the correct direction that are before the intersection
                if vehicle.direction == direction and dist is not None and dist > 0:
                    speed = vehicle.speed if vehicle.speed > 0.1 else 0.1 # Avoid division by zero
                    tti_frames = dist / speed
                    ttis.append(tti_frames / FPS) # Convert to seconds
            
            ttis.sort() # Sort to find the closest (smallest TTI)
            
            # Get the 2 smallest TTIs, using sentinel value if fewer than 2 cars exist
            tti_1 = ttis[0] if len(ttis) > 0 else sentinel_value
            tti_2 = ttis[1] if len(ttis) > 1 else sentinel_value
            features[direction] = {'tti_1': tti_1, 'tti_2': tti_2}
            
        return features

    def _get_state(self) -> np.ndarray:
        """
        Returns the current state representation for the RL agent.

        Returns:
            np.ndarray: State vector of shape (18,)
        """
        queue_lengths = self.sim.get_queue_lengths()
        tti_features = self._get_tti_features()
        intersection_occupancy = self._get_intersection_occupancy_features()

        state = np.array([
            # Predictive Safety Metrics (TTI)
            tti_features['N']['tti_1'], tti_features['N']['tti_2'],
            tti_features['S']['tti_1'], tti_features['S']['tti_2'],
            tti_features['E']['tti_1'], tti_features['E']['tti_2'],
            tti_features['W']['tti_1'], tti_features['W']['tti_2'],
            # Demand Metrics
            queue_lengths['N'],
            queue_lengths['S'],
            queue_lengths['E'],
            queue_lengths['W'],
            # Context Metrics
            self.sim.current_phase,
            self.time_in_current_phase,
            # Immediate Safety Metrics
            intersection_occupancy['N'],
            intersection_occupancy['S'],
            intersection_occupancy['E'],
            intersection_occupancy['W'],
        ], dtype=np.float32)
        return state

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to an initial state and returns an initial observation.
        
        Parameters:
            seed: Optional seed for random number generators
            options: Additional options for resetting the environment (not used)
            
        Returns:
            state (np.ndarray): The initial observation of the environment
            info (dict): Additional information
        """
        super().reset(seed=seed)

        self.sim = Simulation(spawn_configs=self.spawn_configs)

        if self.render_mode == "human": 
            self.gui.simulation = self.sim
            
        self.current_step = 0
        self.collision_count = 0
        self.time_in_current_phase = 0
        
        state = self._get_state()
        info = {}
        return state, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes one time step within the environment.
        
        Parameters:
            action (int): The action taken by the agent (0: Keep, 1: Switch)
            
        Returns:
            state (np.ndarray): The next observation of the environment
            reward (float): The reward received after taking the action
            done (bool): Whether the episode has ended
            truncated (bool): Whether the episode was truncated
            info (dict): Additional information
        """
        # 1. Apply action-based penalties BEFORE the simulation step
        # This directly links the agent's decision to an immediate cost/reward.
        reward = 0
        
        if action == 1: # Agent chose to switch
            reward += REWARD_ACTION_PENALTY

            # NEW: Check for unsafe switching condition
            # Penalize if switching while cars from the currently green direction are in the intersection
            occupancy = self._get_intersection_occupancy_features()
            is_ns_green = self.sim.current_phase == 0
            
            if is_ns_green and (occupancy['N'] > 0 or occupancy['S'] > 0):
                reward += DANGER_ZONE_PENALTY
            elif not is_ns_green and (occupancy['E'] > 0 or occupancy['W'] > 0):
                reward += DANGER_ZONE_PENALTY

            self.sim.control_traffic_lights(action)
            self.time_in_current_phase = 0
        else:
            self.time_in_current_phase += 1

        # 2. Calculate reward over a series of simulation frames
        cumulative_reward = 0
        num_frames = 15
        collision_this_step = False
        for _ in range(num_frames):
            if self.render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        return self._get_state(), 0, True, True, {"info": "Window closed"}
            
            stopped_car_count, cars_passed = self.sim.update()
            
            # --- Non-linear waiting penalty ---
            wait_penalty_this_frame = 0
            for vehicle in self.sim.get_all_vehicles():
                if vehicle.speed < 0.1 and vehicle.wait_time > 0:
                    penalty = (vehicle.wait_time / WAIT_PENALTY_SCALING_FACTOR) ** WAIT_PENALTY_EXPONENT
                    wait_penalty_this_frame += penalty
            
            cumulative_reward -= wait_penalty_this_frame
            # Reward cars that get through
            cumulative_reward += cars_passed * REWARD_THROUGHPUT_BONUS

            if not collision_this_step and self.sim.check_collisions():
                cumulative_reward += REWARD_COLLISION_PENALTY
                self.collision_count += 1
                collision_this_step = True

            if self.render_mode == "human":
                self.gui.draw()
                self.gui.clock.tick(self.metadata["render_fps"])

        self.current_step += 1
        
        # 3. Combine action penalties and simulation rewards
        reward += cumulative_reward

        # 4. Check for truncation (episode length)
        truncated = self.current_step >= MAX_STEPS_PER_EPISODE

        # 5. Info dict and get next state
        info = {}
        state = self._get_state()

        # A collision does not terminate the episode.
        return state, reward, False, truncated, info

    def close(self):
        """
        Closes the environment and any associated resources.
        """
        if self.gui:
            pygame.quit()
            self.gui = None