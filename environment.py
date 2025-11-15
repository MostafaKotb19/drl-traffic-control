# environment.py

from typing import Optional, Dict, Tuple, Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from simulation import Simulation
from gui import GUI
from constants import *

class TrafficEnv(gym.Env):
    """
    A custom Gymnasium environment for reinforcement learning-based traffic light control.

    The agent's goal is to minimize vehicle wait times and prevent collisions by
    deciding when to switch the traffic light phase.
    """
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode: Optional[str] = None, 
                 spawn_rate: Optional[int] = None, 
                 spawn_configs: Optional[Dict[str, int]] = None):
        """
        Initializes the Traffic Environment.

        Args:
            render_mode: If 'human', enables GUI rendering. Otherwise, runs headless.
            spawn_rate: A uniform spawn rate (in frames) for all directions.
            spawn_configs: A dictionary specifying spawn rates per direction, e.g.,
                           {'N': 300, 'S': 400, 'E': 500, 'W': 600}. Overrides `spawn_rate`.
        """
        super(TrafficEnv, self).__init__()
                
        # Set vehicle spawn configuration
        if spawn_configs:
            self.spawn_configs = spawn_configs
        elif spawn_rate:
             self.spawn_configs = {d: spawn_rate for d in ['N', 'S', 'E', 'W']}
        else:
             self.spawn_configs = {d: SPAWN_RATE_NORMAL for d in ['N', 'S', 'E', 'W']}
             
        self.sim = Simulation(spawn_configs=self.spawn_configs)
        
        # Action Space: {0: Keep current phase, 1: Switch to next phase}
        self.action_space = spaces.Discrete(2)
        
        # Observation Space: A vector of 18 features describing the intersection state.
        # - Indices 0-7:  Time-To-Intersection (TTI) for the 2 closest cars in each direction.
        # - Indices 8-11: Queue length (number of stopped cars) for each direction.
        # - Index 12:     Current traffic light phase (0 for NS-green, 1 for EW-green).
        # - Index 13:     Time elapsed in the current phase (in steps).
        # - Indices 14-17: Boolean flags indicating if any car is in the intersection for each direction.
        self.observation_space = spaces.Box(
            low=np.zeros(18, dtype=np.float32),
            high=np.array([
                60, 60,      # Max TTI in seconds for North
                60, 60,      # Max TTI in seconds for South
                60, 60,      # Max TTI in seconds for East
                60, 60,      # Max TTI in seconds for West
                50, 50, 50, 50, # Max queue lengths
                1,              # Max phase index (0 or 1)
                MAX_STEPS_PER_EPISODE, # Max time in current phase
                1, 1, 1, 1      # Intersection occupancy flags
            ], dtype=np.float32),
            dtype=np.float32
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.gui: Optional[GUI] = None
        if self.render_mode == "human":
            self.gui = GUI(self.sim)

        self.collision_count = 0
        self.time_in_current_phase = 0

    def _get_intersection_occupancy_features(self) -> Dict[str, float]:
        """ 
        Checks if any vehicle from a given direction is currently inside the intersection.
        This serves as an immediate safety metric.

        Returns:
            A dictionary mapping each direction to a binary flag (0.0 or 1.0).
        """
        occupancy = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'W': 0.0}
        for vehicle in self.sim.get_all_vehicles():
            if self.sim.intersection_rect.colliderect(vehicle.rect):
                occupancy[vehicle.direction] = 1.0
        return occupancy
    
    def _get_tti_features(self) -> Dict[str, Dict[str, float]]:
        """
        Calculates the Time-To-Intersection (TTI) in seconds for the two closest
        approaching vehicles in each direction. This is a predictive safety metric.

        Returns:
            A nested dictionary with TTI values for each direction.
            e.g., {'N': {'tti_1': 10.5, 'tti_2': 25.0}, ...}
        """
        # A high TTI value is used as a sentinel if fewer than two cars are approaching.
        sentinel_value = 60.0 
        features = {}

        for direction in ['N', 'S', 'E', 'W']:
            ttis = []
            for vehicle in self.sim.get_all_vehicles():
                dist = self.sim._get_dist_to_intersection(vehicle)
                # Consider only vehicles of the correct direction that are before the intersection.
                if vehicle.direction == direction and dist is not None and dist > 0:
                    speed = vehicle.speed if vehicle.speed > 0.1 else 0.1 # Avoid division by zero
                    tti_frames = dist / speed
                    ttis.append(tti_frames / FPS) # Convert from frames to seconds
            
            ttis.sort() # Sort to find the closest vehicles (smallest TTI)
            
            # Populate features with the two smallest TTIs, using the sentinel value if needed.
            tti_1 = ttis[0] if len(ttis) > 0 else sentinel_value
            tti_2 = ttis[1] if len(ttis) > 1 else sentinel_value
            features[direction] = {'tti_1': tti_1, 'tti_2': tti_2}
            
        return features

    def _get_state(self) -> np.ndarray:
        """
        Constructs the current state observation vector for the RL agent.

        Returns:
            A numpy array of shape (18,) representing the full state.
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
            # Demand Metrics (Queue Lengths)
            queue_lengths['N'],
            queue_lengths['S'],
            queue_lengths['E'],
            queue_lengths['W'],
            # Context Metrics
            self.sim.current_phase,
            self.time_in_current_phase,
            # Immediate Safety Metrics (Intersection Occupancy)
            intersection_occupancy['N'],
            intersection_occupancy['S'],
            intersection_occupancy['E'],
            intersection_occupancy['W'],
        ], dtype=np.float32)
        return state

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to its initial state.

        Args:
            seed: Optional seed for the random number generator.
            options: Optional dictionary for additional reset options (unused).
            
        Returns:
            A tuple containing the initial observation and an empty info dictionary.
        """
        super().reset(seed=seed)

        # Re-initialize the simulation to start a new episode
        self.sim = Simulation(spawn_configs=self.spawn_configs)

        if self.render_mode == "human": 
            if self.gui is None: # Re-initialize GUI if it was closed
                self.gui = GUI(self.sim)
            self.gui.simulation = self.sim
            
        self.current_step = 0
        self.collision_count = 0
        self.time_in_current_phase = 0
        
        state = self._get_state()
        info = {}
        return state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes one time step in the environment.

        An agent "step" consists of multiple simulation frames (e.g., 15 frames)
        to allow for meaningful state changes and reward accumulation.

        Args:
            action: The action taken by the agent (0: Keep, 1: Switch).
            
        Returns:
            A tuple containing the next state, reward, terminated flag, truncated flag, and info dict.
        """
        # 1. Apply immediate penalties based on the agent's action
        reward = 0.0
        
        if action == 1: # Agent chose to switch phases
            reward += REWARD_ACTION_PENALTY

            # Penalize switching if cars from the currently green direction are in the intersection
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

        # 2. Run the simulation for a set number of frames and accumulate rewards
        cumulative_reward = 0.0
        num_frames = 15  # Each agent step corresponds to 15 simulation frames
        collision_this_step = False
        for _ in range(num_frames):
            if self.render_mode == "human" and self.gui:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        # If the user closes the window, end the episode
                        self.close()
                        return self._get_state(), 0, True, True, {"info": "Window closed"}
            
            stopped_car_count, cars_passed = self.sim.update()
            
            # Apply a non-linear penalty for waiting time to heavily discourage long queues
            wait_penalty_this_frame = 0
            for vehicle in self.sim.get_all_vehicles():
                if vehicle.speed < 0.1 and vehicle.wait_time > 0:
                    penalty = (vehicle.wait_time / WAIT_PENALTY_SCALING_FACTOR) ** WAIT_PENALTY_EXPONENT
                    wait_penalty_this_frame += penalty
            
            cumulative_reward -= wait_penalty_this_frame
            cumulative_reward += cars_passed * REWARD_THROUGHPUT_BONUS

            # Apply a large penalty for collisions
            if not collision_this_step and self.sim.check_collisions():
                cumulative_reward += REWARD_COLLISION_PENALTY
                self.collision_count += 1
                collision_this_step = True

            if self.render_mode == "human" and self.gui:
                self.gui.draw()
                self.gui.clock.tick(self.metadata["render_fps"])

        self.current_step += 1
        
        # 3. Combine action penalties and simulation rewards
        reward += cumulative_reward

        # 4. Check for truncation (episode reached max length)
        truncated = self.current_step >= MAX_STEPS_PER_EPISODE

        # 5. Get next state and return
        info = {}
        state = self._get_state()

        # An episode is not terminated by a collision, only by truncation.
        return state, reward, False, truncated, info

    def close(self):
        """Closes the environment and any associated resources, like the Pygame window."""
        if self.gui:
            pygame.quit()
            self.gui = None