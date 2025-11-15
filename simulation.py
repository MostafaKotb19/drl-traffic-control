# simulation.py

import pygame
import random
from vehicle import Vehicle
from traffic_light import TrafficLight
from road import Lane
from constants import *

class Simulation:
    def __init__(self, spawn_configs: dict):
        """
        Initializes the traffic simulation.
        
        Parameters:
            spawn_configs: Dictionary with spawn rates for each direction {'N': rate, 'S': rate, 'E': rate, 'W': rate}
        """
        self.spawn_configs = spawn_configs
        center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
        self.intersection_rect = pygame.Rect(
            center_x - ROAD_WIDTH, center_y - ROAD_WIDTH, ROAD_WIDTH * 2, ROAD_WIDTH * 2
        )
        
        self.lanes = self._create_lanes()
        self.traffic_lights = self._create_traffic_lights()
        self.time_since_last_spawn = {d: 0 for d in ['N', 'S', 'E', 'W']}
        self.next_spawn_interval = {d: self._get_next_spawn_interval(d) for d in ['N', 'S', 'E', 'W']}
        
        self.current_phase = 0 # 0: NS green, 1: EW green
        self.set_traffic_light_state()

        self.cars_passed_intersection = 0
        self.total_wait_time_steps = 0 # Total steps cars have been waiting
        self.completed_car_wait_times = []

    def _get_next_spawn_interval(self, direction: str) -> int:
        """
        Calculates a randomized next spawn time in frames.
        
        Parameters:
            direction: 'N', 'S', 'E', or 'W'
        
        Returns:
            int: Number of frames until the next vehicle spawns in this direction
        """
        rate = self.spawn_configs.get(direction, 9999) # Get spawn rate in frames
        # Add jitter: spawn time will be between 50% and 150% of the base rate
        return int(rate * (0.5 + random.random() * 1.0))
    
    def _create_lanes(self) -> dict:
        """
        Creates the lanes for the intersection.

        Returns:
            dict: A dictionary of Lane objects keyed by direction and lane index.
        """
        lanes = {}
        center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
        
        lanes['N_0'] = Lane((center_x - LANE_WIDTH/2, SCREEN_HEIGHT), (center_x - LANE_WIDTH/2, 0), 'N', 0)
        lanes['N_1'] = Lane((center_x - LANE_WIDTH*1.5, SCREEN_HEIGHT), (center_x - LANE_WIDTH*1.5, 0), 'N', 1)
        lanes['S_0'] = Lane((center_x + LANE_WIDTH/2, 0), (center_x + LANE_WIDTH/2, SCREEN_HEIGHT), 'S', 0)
        lanes['S_1'] = Lane((center_x + LANE_WIDTH*1.5, 0), (center_x + LANE_WIDTH*1.5, SCREEN_HEIGHT), 'S', 1)
        lanes['W_0'] = Lane((SCREEN_WIDTH, center_y + LANE_WIDTH/2), (0, center_y + LANE_WIDTH/2), 'W', 0)
        lanes['W_1'] = Lane((SCREEN_WIDTH, center_y + LANE_WIDTH*1.5), (0, center_y + LANE_WIDTH*1.5), 'W', 1)
        lanes['E_0'] = Lane((0, center_y - LANE_WIDTH/2), (SCREEN_WIDTH, center_y - LANE_WIDTH/2), 'E', 0)
        lanes['E_1'] = Lane((0, center_y - LANE_WIDTH*1.5), (SCREEN_WIDTH, center_y - LANE_WIDTH*1.5), 'E', 1)

        return lanes
    
    def _create_traffic_lights(self) -> dict:
        """
        Creates the traffic lights for the intersection.

        Returns:
            dict: A dictionary of TrafficLight objects keyed by direction.
        """
        lights = {}
        center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
        offset = 15
        
        lights['N'] = TrafficLight(center_x - ROAD_WIDTH/2, center_y + ROAD_WIDTH - offset)
        lights['S'] = TrafficLight(center_x + ROAD_WIDTH/2, center_y - ROAD_WIDTH + offset)
        lights['W'] = TrafficLight(center_x + ROAD_WIDTH - offset, center_y + ROAD_WIDTH/2)
        lights['E'] = TrafficLight(center_x - ROAD_WIDTH + offset, center_y - ROAD_WIDTH/2)
        return lights

    def set_traffic_light_state(self):
        """
        Sets the traffic light states based on the current phase.
        """
        ns_state = 'green' if self.current_phase == 0 else 'red'
        self.traffic_lights['N'].set_state(ns_state)
        self.traffic_lights['S'].set_state(ns_state)
        ew_state = 'green' if self.current_phase == 1 else 'red'
        self.traffic_lights['E'].set_state(ew_state)
        self.traffic_lights['W'].set_state(ew_state)

    def control_traffic_lights(self, action: int):
        """
        Controls the traffic lights based on the given action.

        Parameters:
            action (int): 0 to keep lights the same, 1 to switch lights.
        """
        # Action 0: Keep lights the same
        # Action 1: Switch lights
        if action == 1 and self.current_phase == 0:
            self.current_phase = 1
        elif action == 1 and self.current_phase == 1:
            self.current_phase = 0
        
        self.set_traffic_light_state()
    
    def update(self) -> tuple:
        """
        Updates the simulation state by one timestep.
        
        Returns:
            int: Number of cars stopped this frame
            int: Number of cars that passed the intersection this frame
        """
        # Spawn vehicles based on per-direction timers and rates
        for direction in self.time_since_last_spawn.keys():
            self.time_since_last_spawn[direction] += 1
            if self.time_since_last_spawn[direction] >= self.next_spawn_interval[direction]:
                self._spawn_vehicle(direction)
                self.time_since_last_spawn[direction] = 0
                self.next_spawn_interval[direction] = self._get_next_spawn_interval(direction)
        
        # We will return these values to the environment for reward calculation
        stopped_car_count_this_frame = 0

        # Accumulate total system wait time for evaluation metrics (this is different from the reward)
        for vehicle in self.get_all_vehicles():
            if vehicle.stopped and vehicle.speed < 0.1:
                self.total_wait_time_steps += 1

        # Update all vehicles
        for lane in self.lanes.values():
            for vehicle in lane.vehicles:
                lead_vehicle = lane.get_lead_vehicle(vehicle)
                light = self.traffic_lights[vehicle.direction]
                dist_to_intersection = self._get_dist_to_intersection(vehicle)
                is_before_intersection = dist_to_intersection is not None and dist_to_intersection > 0
                is_light_red = light.state == 'red'
                is_in_braking_zone = is_before_intersection and dist_to_intersection < 35
                final_stop_signal = is_light_red and is_in_braking_zone
                vehicle.update(lead_vehicle, final_stop_signal)

                if vehicle.stopped and vehicle.speed < 0.1:
                    stopped_car_count_this_frame += 1

        # Remove vehicles that are out of bounds and count them, storing their wait times
        cars_passed_this_frame = 0
        for lane in self.lanes.values():
            surviving_vehicles = []
            for vehicle in lane.vehicles:
                if self._is_in_bounds(vehicle):
                    surviving_vehicles.append(vehicle)
                else:
                    # Vehicle has passed, record its stats before removing
                    self.completed_car_wait_times.append(vehicle.total_wait_time)
                    cars_passed_this_frame += 1
            lane.vehicles = surviving_vehicles

        self.cars_passed_intersection += cars_passed_this_frame

        return stopped_car_count_this_frame, cars_passed_this_frame

    def _get_dist_to_intersection(self, vehicle: Vehicle) -> float:
        """
        Returns the distance from the vehicle to the intersection stop line.
        
        Parameters:
            vehicle: The vehicle object.
        
        Returns:
            float: Distance in pixels to the intersection stop line, or None if not applicable.
        """
        stop_line_margin = 5
        if vehicle.direction == 'N':
            return vehicle.rect.top - (self.intersection_rect.bottom + stop_line_margin)
        if vehicle.direction == 'S':
            return (self.intersection_rect.top - stop_line_margin) - vehicle.rect.bottom
        if vehicle.direction == 'W':
            return vehicle.rect.left - (self.intersection_rect.right + stop_line_margin)
        if vehicle.direction == 'E':
            return (self.intersection_rect.left - stop_line_margin) - vehicle.rect.right
        return None

    def _spawn_vehicle(self, direction: str):
        """
        Spawns a vehicle in the specified direction if possible.

        Parameters:
            direction: 'N', 'S', 'E', or 'W'
        """
        # Find lanes corresponding to the given direction
        spawn_lanes = [lane for lane in self.lanes.values() if lane.direction == direction]
        if not spawn_lanes: return
        
        lane = random.choice(spawn_lanes)
        x, y = lane.start_pos
        
        # Initial position adjustment to be off-screen
        if lane.direction == 'N': y = SCREEN_HEIGHT
        elif lane.direction == 'S': y = 0 - CAR_LENGTH
        elif lane.direction == 'W': x = SCREEN_WIDTH
        elif lane.direction == 'E': x = 0 - CAR_LENGTH
        
        new_vehicle = Vehicle(x, y, lane.direction, lane.lane_index)

        # Ensure we don't spawn a car on top of another one
        if not lane.vehicles or new_vehicle.get_distance_to(lane.vehicles[-1]) > 0:
            lane.vehicles.append(new_vehicle)

    def _is_in_bounds(self, vehicle: Vehicle) -> bool:
        """
        Checks if the vehicle is still within the simulation bounds.
        
        Parameters:
            vehicle: The vehicle object.
            
        Returns:
            bool: True if in bounds, False if out of bounds.
        """
        if vehicle.direction == 'N' and vehicle.rect.bottom < 0: return False
        if vehicle.direction == 'S' and vehicle.rect.top > SCREEN_HEIGHT: return False
        if vehicle.direction == 'W' and vehicle.rect.right < 0: return False
        if vehicle.direction == 'E' and vehicle.rect.left > SCREEN_WIDTH: return False
        return True
    
    def get_queue_lengths(self) -> dict:
        """
        Counts the number of stopped vehicles approaching the intersection for each direction.
        
        Returns:
            dict: A dictionary with queue lengths for each direction {'N': int, 'S': int, 'E': int, 'W': int}
        """
        queue_lengths = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        for vehicle in self.get_all_vehicles():
            dist_to_intersection = self._get_dist_to_intersection(vehicle)
            # A car is in a queue if it's stopped before the intersection
            if vehicle.speed < 0.1 and dist_to_intersection is not None and dist_to_intersection > 0:
                queue_lengths[vehicle.direction] += 1
        return queue_lengths
    
    def get_observation_info(self) -> dict:
        """
        Returns the total accumulated wait time for cars in each approach.

        Returns:
            dict: A dictionary with total wait times for each direction {'N': int, 'S': int, 'E': int, 'W': int}
        """
        wait_times = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
        for vehicle in self.get_all_vehicles():
            is_before_intersection = self._get_dist_to_intersection(vehicle)
            if is_before_intersection is not None and is_before_intersection > 0:
                wait_times[vehicle.direction] += vehicle.wait_time
        return wait_times

    def get_system_metrics(self) -> tuple:
        """
        Calculates system-wide metrics for the reward function.
        
        Returns:
            tuple: (average_speed: float, total_jerk: float)
        """
        all_vehicles = self.get_all_vehicles()
        if not all_vehicles:
            return 0, 0

        total_speed = sum(v.speed for v in all_vehicles)
        avg_speed = total_speed / len(all_vehicles)
        
        # "Jerk" is proxied by the sum of absolute speed changes
        total_jerk = sum(abs(v.speed - v.last_speed) for v in all_vehicles)
        
        return avg_speed, total_jerk

    def check_collisions(self) -> bool:
        """
        Checks for collisions between perpendicular vehicles in the intersection.
        
        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        vehicles_in_intersection = []
        for lane in self.lanes.values():
            for vehicle in lane.vehicles:
                if self.intersection_rect.colliderect(vehicle.rect):
                    vehicles_in_intersection.append(vehicle)
        
        for i in range(len(vehicles_in_intersection)):
            for j in range(i + 1, len(vehicles_in_intersection)):
                v1 = vehicles_in_intersection[i]
                v2 = vehicles_in_intersection[j]
                
                # Correctly check if one vehicle is on a vertical path and the other is on a horizontal path.
                is_v1_vertical = v1.direction in ['N', 'S']
                is_v2_vertical = v2.direction in ['N', 'S']

                if is_v1_vertical != is_v2_vertical: # This ensures they are on perpendicular paths
                    if v1.rect.colliderect(v2.rect):
                        return True # Collision detected
        return False
    
    def get_all_vehicles(self) -> list:
        """
        Returns a single list of all vehicles in the simulation.
        
        Returns:
            list: List of all Vehicle objects.
        """
        all_vehicles = []
        for lane in self.lanes.values():
            all_vehicles.extend(lane.vehicles)
        return all_vehicles