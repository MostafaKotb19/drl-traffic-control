# vehicle.py

import pygame
import random
from constants import *

class Vehicle:
    def __init__(self, x: int, y: int, direction: str, lane_index: int):
        """
        Represents a vehicle in the traffic simulation.
        
        Parameters:
            x (int): X-coordinate of the vehicle
            y (int): Y-coordinate of the vehicle
            direction (str): Direction of travel ('N', 'S', 'E', 'W')
            lane_index (int): Index of the lane (0 for left lane, 1 for right lane)
        """
        self.direction = direction  # 'N', 'S', 'E', 'W'
        self.lane_index = lane_index # 0 or 1
        
        # Determine dimensions based on direction
        if self.direction in ['N', 'S']:
            self.width = CAR_WIDTH
            self.height = CAR_LENGTH
        else: # 'E', 'W'
            self.width = CAR_LENGTH
            self.height = CAR_WIDTH
            
        # The Rect is now the primary object for position and collision
        self.rect = pygame.Rect(x, y, self.width, self.height)
        
        # We use float versions of x/y for precise speed/acceleration calculations
        self.x_float = float(x)
        self.y_float = float(y)

        self.speed = 0
        self.last_speed = 0
        self.max_speed = MAX_SPEED + random.uniform(-MAX_SPEED_VARIATION, MAX_SPEED_VARIATION)
        self.acceleration = ACCELERATION
        self.color = (0, 150, 255) # Blue
        self.stopped = False
        self.wait_time = 0
        self.total_wait_time = 0

    def update(self, lead_vehicle, traffic_light_is_red: bool):
        """
        Updates the vehicle's position and speed based on lead vehicle and traffic light.
        
        Parameters:
            lead_vehicle (Vehicle): The vehicle directly ahead in the same lane (or None)
            traffic_light_is_red (bool): Whether the traffic light ahead is red
        """
        self.last_speed = self.speed
        
        # Rule 1: Check for traffic light
        if traffic_light_is_red:
            self.stopped = True
        # Rule 2: Check for lead vehicle
        elif lead_vehicle:
            distance = self.get_distance_to(lead_vehicle)
            if distance < SAFE_DISTANCE:
                self.stopped = True
            else:
                self.stopped = False
        else:
            self.stopped = False

        # Update speed based on stopped status
        if self.stopped:
            self.speed = max(0, self.speed - BRAKING_DECELERATION)
        else:
            self.speed = min(self.max_speed, self.speed + self.acceleration)
            
        # If stopped, accumulate wait time. If moving, reset wait time.
        if self.stopped and self.speed < 0.1:
            self.wait_time += 1
            self.total_wait_time += 1
        else:
            self.wait_time = 0

        # Update precise float positions
        if self.direction == 'N':
            self.y_float -= self.speed
        elif self.direction == 'S':
            self.y_float += self.speed
        elif self.direction == 'W':
            self.x_float -= self.speed
        elif self.direction == 'E':
            self.x_float += self.speed
        
        # Update the integer Rect position from the float values
        self.rect.x = int(self.x_float)
        self.rect.y = int(self.y_float)

    def get_distance_to(self, other_vehicle) -> float:
        """
        Returns the distance to another vehicle in the direction of travel.

        Parameters:
            other_vehicle (Vehicle): The other vehicle to measure distance to.
        
        Returns:
            float: The distance to the other vehicle in the direction of travel.
        """
        # Use rect properties for cleaner, more accurate distance calculation
        if self.direction == 'N':
            return self.rect.top - other_vehicle.rect.bottom
        elif self.direction == 'S':
            return other_vehicle.rect.top - self.rect.bottom
        elif self.direction == 'W':
            return self.rect.left - other_vehicle.rect.right
        elif self.direction == 'E':
            return other_vehicle.rect.left - self.rect.right
        return float('inf')

    def draw(self, surface: pygame.Surface):
        """
        Draws the vehicle on the given surface.
        """
        # Draw the vehicle's rect directly
        pygame.draw.rect(surface, self.color, self.rect)