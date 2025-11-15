import pygame
from constants import *

class GUI:
    def __init__(self, simulation):
        """
        Initializes the GUI with the given simulation instance.
        
        Parameters:
            simulation: An instance of the TrafficSimulation class.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Traffic Simulation")
        self.simulation = simulation
        self.clock = pygame.time.Clock()

    def run(self):
        """
        Main loop to run the GUI.
        """
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.simulation.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()

    def draw(self):
        """
        Draws the current state of the simulation onto the screen.
        """
        # Background
        self.screen.fill(LIGHT_GREEN)
        
        center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2

        # Draw the roads (as two big rectangles)
        pygame.draw.rect(self.screen, DARK_GREY, (center_x - ROAD_WIDTH, 0, ROAD_WIDTH * 2, SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, DARK_GREY, (0, center_y - ROAD_WIDTH, SCREEN_WIDTH, ROAD_WIDTH * 2))

        # Draw Road Dividers (thick yellow lines)
        pygame.draw.line(self.screen, YELLOW, (center_x, 0), (center_x, center_y - ROAD_WIDTH), ROAD_DIVIDER_WIDTH)
        pygame.draw.line(self.screen, YELLOW, (center_x, center_y + ROAD_WIDTH), (center_x, SCREEN_HEIGHT), ROAD_DIVIDER_WIDTH)
        pygame.draw.line(self.screen, YELLOW, (0, center_y), (center_x - ROAD_WIDTH, center_y), ROAD_DIVIDER_WIDTH)
        pygame.draw.line(self.screen, YELLOW, (center_x + ROAD_WIDTH, center_y), (SCREEN_WIDTH, center_y), ROAD_DIVIDER_WIDTH)

        # Draw Lane Dividers (dashed white lines)
        self._draw_dashed_line_set()
            
        # Vehicles
        for lane in self.simulation.lanes.values():
            for vehicle in lane.vehicles:
                vehicle.draw(self.screen)
        
        # Traffic Lights
        for light in self.simulation.traffic_lights.values():
            light.draw(self.screen)
            
        pygame.display.flip()
        
    def _draw_dashed_line(self, start_pos: tuple, end_pos: tuple):
        """
        Draws a dashed line between two points.
        
        Parameters:
            start_pos: Tuple (x, y) for the start position.
            end_pos: Tuple (x, y) for the end position.
        """
        dash_length = 10
        gap_length = 10
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = (dx**2 + dy**2)**0.5
        if distance == 0: return # Avoid division by zero
        dx, dy = dx / distance, dy / distance 
        
        current_pos = list(start_pos)
        drawn_length = 0
        
        while drawn_length < distance:
            draw_end_x = current_pos[0] + dx * dash_length
            draw_end_y = current_pos[1] + dy * dash_length
            pygame.draw.line(self.screen, WHITE, current_pos, (draw_end_x, draw_end_y), LANE_DIVIDER_WIDTH)
            
            current_pos[0] += dx * (dash_length + gap_length)
            current_pos[1] += dy * (dash_length + gap_length)
            drawn_length += dash_length + gap_length

    def _draw_dashed_line_set(self):
        """
        Draws all the dashed lane dividers.
        """
        center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
        
        # Vertical Road lanes
        self._draw_dashed_line((center_x - LANE_WIDTH, SCREEN_HEIGHT), (center_x - LANE_WIDTH, center_y + ROAD_WIDTH))
        self._draw_dashed_line((center_x - LANE_WIDTH, center_y - ROAD_WIDTH), (center_x - LANE_WIDTH, 0))
        self._draw_dashed_line((center_x + LANE_WIDTH, 0), (center_x + LANE_WIDTH, center_y - ROAD_WIDTH))
        self._draw_dashed_line((center_x + LANE_WIDTH, center_y + ROAD_WIDTH), (center_x + LANE_WIDTH, SCREEN_HEIGHT))

        # Horizontal Road lanes
        self._draw_dashed_line((SCREEN_WIDTH, center_y + LANE_WIDTH), (center_x + ROAD_WIDTH, center_y + LANE_WIDTH))
        self._draw_dashed_line((center_x - ROAD_WIDTH, center_y + LANE_WIDTH), (0, center_y + LANE_WIDTH))
        self._draw_dashed_line((0, center_y - LANE_WIDTH), (center_x - ROAD_WIDTH, center_y - LANE_WIDTH))
        self._draw_dashed_line((center_x + ROAD_WIDTH, center_y - LANE_WIDTH), (SCREEN_WIDTH, center_y - LANE_WIDTH))