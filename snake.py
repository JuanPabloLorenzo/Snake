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
        
        self.body.insert(0, new_head)
        if new_head != food_position:
            self.body.pop()

    def change_direction(self, new_direction):
        # Don't let the snake go in the opposite direction
        # if self.direction == (new_direction + 2) % 4:
        #     return
        
        self.direction = new_direction
