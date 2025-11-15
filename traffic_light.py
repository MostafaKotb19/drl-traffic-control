# traffic_light.py

import pygame
from typing import Literal
from constants import *

LightState = Literal['red', 'yellow', 'green']

class TrafficLight:
    def __init__(self, x: int, y: int):
        """
        Represents a single traffic light bulb in the simulation.

        Args:
            x: The x-coordinate of the light's center.
            y: The y-coordinate of the light's center.
        """
        self.x = x
        self.y = y
        self.state: LightState = 'red'
        self.colors = {
            'red': RED,
            'yellow': YELLOW,
            'green': GREEN
        }

    def set_state(self, state: LightState):
        """
        Sets the color state of the traffic light.

        Args:
            state: The new state, must be one of 'red', 'yellow', or 'green'.
        """
        if state in self.colors:
            self.state = state
        else:
            raise ValueError(f"Invalid light state: {state}")

    def draw(self, surface: pygame.Surface):
        """
        Draws the traffic light on the given Pygame surface.

        Args:
            surface: The Pygame surface to draw on.
        """
        color = self.colors[self.state]
        # Draw a black border around the light for better visibility
        pygame.draw.circle(surface, BLACK, (self.x, self.y), LIGHT_RADIUS + 2)
        pygame.draw.circle(surface, color, (self.x, self.y), LIGHT_RADIUS)