import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from snake import Snake
import random
import time
import sys
import numpy as np
import tensorflow as tf
from scene import Scene

# Normal scene but using variable length snake
# The length will be determined by a normal distribution
class SceneLongerSnake(Scene):
    def __init__(self, init_randomly=False, map_height=10, map_width=10, block_size=15, snake_longer_prob=0.5, length_mean=10, length_std=2):
        self.snake_longer_prob = snake_longer_prob
        self.length_mean = length_mean
        self.length_std = length_std
        super().__init__(init_randomly, map_height, map_width, block_size)
    
    def init_snake(self):
        rand_num = random.random()
        if rand_num < self.snake_longer_prob:
            temp_initial_length = int(np.random.normal(self.length_mean, self.length_std))
            temp_initial_length = min(temp_initial_length, self.width * self.height - 1)
            temp_initial_length = max(temp_initial_length, 2)
        else:
            temp_initial_length = 2
        
        if self.init_randomly:
            head = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            
            possible = [0, 1, 2, 3]
            if head[0] == 0:
                possible.remove(0)
            if head[0] == self.width - 1:
                possible.remove(2)
            if head[1] == 0:
                possible.remove(1)
            if head[1] == self.height - 1:
                possible.remove(3)
            
            tail = random.choice(possible)
            snake_initial_body = [head]
            if tail == 0:
                snake_initial_body.append((head[0] - 1, head[1]))
            elif tail == 1:
                snake_initial_body.append((head[0], head[1] - 1))
            elif tail == 2:
                snake_initial_body.append((head[0] + 1, head[1]))
            elif tail == 3:
                snake_initial_body.append((head[0], head[1] + 1))
                
            self.snake = Snake(snake_initial_body)
            
            for i in range(1, temp_initial_length - 1):
                possible = [0, 1, 2, 3]
                if self.snake.body[i][0] == 0 or (self.snake.body[i][0] - 1, self.snake.body[i][1]) in self.snake.body:
                    possible.remove(0)
                if self.snake.body[i][0] == self.width - 1 or (self.snake.body[i][0] + 1, self.snake.body[i][1]) in self.snake.body:
                    possible.remove(2)
                if self.snake.body[i][1] == 0 or (self.snake.body[i][0], self.snake.body[i][1] - 1) in self.snake.body:
                    possible.remove(1)
                if self.snake.body[i][1] == self.height - 1 or (self.snake.body[i][0], self.snake.body[i][1] + 1) in self.snake.body:
                    possible.remove(3)
                
                if len(possible) == 0:
                    break
                
                tail = random.choice(possible)
                snake_initial_body = [head]
                if tail == 0:
                    self.snake.body.append((self.snake.body[i][0] - 1, self.snake.body[i][1]))
                elif tail == 1:
                    self.snake.body.append((self.snake.body[i][0], self.snake.body[i][1] - 1))
                elif tail == 2:
                    self.snake.body.append((self.snake.body[i][0] + 1, self.snake.body[i][1]))
                elif tail == 3:
                    self.snake.body.append((self.snake.body[i][0], self.snake.body[i][1] + 1))
                    
            
            # Direction
            possible_directions = [0, 1, 2, 3]
            if head[0] == 0 or (head[0] - 1, head[1]) in self.snake.body:
                possible_directions.remove(0)
            if head[0] == self.width - 1 or (head[0] + 1, head[1]) in self.snake.body:
                possible_directions.remove(2)
            if head[1] == 0 or (head[0], head[1] - 1) in self.snake.body:
                possible_directions.remove(1)
            if head[1] == self.height - 1 or (head[0], head[1] + 1) in self.snake.body:
                possible_directions.remove(3)
                
            if len(possible_directions) == 0:
                direction = random.choice([0, 1, 2, 3])
            else:
                direction = random.choice(possible_directions)
                
            self.snake.direction = direction
        else:
            snake_initial_body = [(self.width // 2, self.height // 2), (self.width // 2 + 1, self.height // 2)]
            direction = 0
            self.snake = Snake(snake_initial_body)
            self.snake.direction = direction
        

        self.snake.direction = direction
            
if __name__ == '__main__':
    scene = SceneLongerSnake(init_randomly=True, snake_longer_prob=0.5, length_mean=10, length_std=2)
    scene.run()
