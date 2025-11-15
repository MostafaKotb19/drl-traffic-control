# road.py

import pygame
from constants import *
from vehicle import Vehicle

class Lane:
    def __init__(self, start_pos: tuple, end_pos: tuple, direction: str, lane_index: int):
        """
        Represents a single lane on a road.
        
        Parameters:
            start_pos: (x, y) tuple for the start of the lane
            end_pos: (x, y) tuple for the end of the lane
            direction: 'N', 'S', 'E', 'W' indicating lane direction
            lane_index: Index of the lane (0 for left lane, 1 for right lane)
        """
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.direction = direction # 'N', 'S', 'E', 'W'
        self.lane_index = lane_index
        self.vehicles = []

    def get_lead_vehicle(self, vehicle: Vehicle):
        """
        Returns the vehicle directly ahead in this lane, if any.
        
        Parameters:
            vehicle: The vehicle for which to find the lead vehicle.
        
        Returns:
            vehicle: The lead vehicle object or None if no lead vehicle exists."""
        try:
            vehicle_idx = self.vehicles.index(vehicle)
            if vehicle_idx > 0:
                return self.vehicles[vehicle_idx - 1]
        except ValueError:
            # Vehicle not in this lane
            pass
        return None

    def draw(self, surface: pygame.Surface):
        """
        Draws the lane on the given surface.
        
        Parameters:
            surface: The pygame surface to draw on.
            """
        pygame.draw.line(surface, DARK_GREY, self.start_pos, self.end_pos, int(ROAD_WIDTH))
