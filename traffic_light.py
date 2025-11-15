import pygame
from constants import *

class TrafficLight:
    def __init__(self, x: int, y: int):
        """
        Represents a traffic light at a given position.
        
        Parameters:
            x (int): X-coordinate of the traffic light
            y (int): Y-coordinate of the traffic light
        """
        self.x = x
        self.y = y
        self.state = 'red' # 'red', 'yellow', 'green'
        self.colors = {
            'red': RED,
            'yellow': YELLOW,
            'green': GREEN
        }

    def set_state(self, state: str):
        """
        Sets the state of the traffic light.

        Parameters:
            state (str): New state ('red', 'yellow', 'green')
        """
        self.state = state

    def draw(self, surface: pygame.Surface):
        """
        Draws the traffic light on the given surface.

        Parameters:
            surface (pygame.Surface): The pygame surface to draw on.
        """
        color = self.colors[self.state]
        pygame.draw.circle(surface, BLACK, (self.x, self.y), LIGHT_RADIUS + 2) # Border
        pygame.draw.circle(surface, color, (self.x, self.y), LIGHT_RADIUS)