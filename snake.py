import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import random

class Snake:
    def __init__(self, initial_body):
        self.body = initial_body
        # Directions: {0:left, 1:up, 2:right, 3:down}
        self.direction = 0

    def move(self, food_position):
        head = self.body[0]
        if self.direction == 0:
            new_head = (head[0] - 1, head[1])
        elif self.direction == 1:
            new_head = (head[0], head[1] - 1)
        elif self.direction == 2:
            new_head = (head[0] + 1, head[1])
        elif self.direction == 3:
            new_head = (head[0], head[1] + 1)
            
        # If the new head collides with the second element of the body, return false
        if new_head == self.body[1]:
            return False
        
        self.body.insert(0, new_head)
        if new_head != food_position:
            self.body.pop()
            
        if self.body[0] in self.body[1:]:
            return False
        
        return True

    def change_direction(self, new_direction):
        self.direction = new_direction
