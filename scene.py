import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from snake import Snake
import random
import time
import sys
import numpy as np
import tensorflow as tf

MAP_WIDTH = 5
MAP_HEIGHT = 5
BLOCK_SIZE = 15

class Scene:
    def __init__(self, init_randomly=False):
        self.height = MAP_HEIGHT; self.width = MAP_WIDTH
        self.block_size = BLOCK_SIZE
        self.init_randomly = init_randomly
        self.reset()
        self.elements_count = 4 # Empty, Snake Head, Snake Body, Food
        
    def move(self, action):
        self.snake.change_direction(action)
        done = False
        prev_head = self.snake.body[0]
        valid_move = self.snake.move(self.food_position)
        head_x, head_y = self.snake.body[0]
        
        # The snake collided with itself or move in the opposite direction or collided with the wall
        if (not valid_move) or head_x < 0 or head_x >= self.width or head_y < 0 or head_y >= self.height:
            done = True
            reward = -20
        elif self.snake.body[0] == self.food_position:
            done = len(self.snake.body) == self.width * self.height
            if done:
                reward = 100
            else:
                reward = 5
                self.new_food_position()
        else:
            reward = 0
            
        if done:
            self.init_snake()
            self.new_food_position()
            
        next_state = self.state()
        
        return next_state, reward, done
    
    def init_snake(self):
        if self.init_randomly:
            head = (random.randint(0, MAP_WIDTH - 1), random.randint(0, MAP_HEIGHT - 1))
            
            possible = [0, 1, 2, 3]
            if head[0] == 0:
                possible.remove(0)
            if head[0] == MAP_WIDTH - 1:
                possible.remove(2)
            if head[1] == 0:
                possible.remove(1)
            if head[1] == MAP_HEIGHT - 1:
                possible.remove(3)
            
            tail = random.choice(possible)
            possible.remove(tail)
            direction = random.choice(possible)
                
            snake_initial_body = [head]
            if tail == 0:
                snake_initial_body.append((head[0] - 1, head[1]))
            elif tail == 1:
                snake_initial_body.append((head[0], head[1] - 1))
            elif tail == 2:
                snake_initial_body.append((head[0] + 1, head[1]))
            elif tail == 3:
                snake_initial_body.append((head[0], head[1] + 1))
        else:
            snake_initial_body = [(MAP_WIDTH // 2, MAP_HEIGHT // 2), (MAP_WIDTH // 2 + 1, MAP_HEIGHT // 2)]
            direction = 0
        
        self.snake = Snake(snake_initial_body)
        self.snake.direction = direction
            
    def new_food_position(self):
        possible_food_positions = [(x, y) for x in range(MAP_WIDTH) for y in range(MAP_HEIGHT) if (x, y) not in self.snake.body]
        self.food_position = random.choice(possible_food_positions)
        
    def reset(self):
        self.init_snake()
        self.new_food_position()
        
    def draw(self):
        self.screen.fill((255, 255, 255))
        
        # Draw the snake
        for block in self.snake.body:
            pygame.draw.rect(self.screen, (0, 0, 0), (block[0] * BLOCK_SIZE, block[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            
        # Draw the food
        pygame.draw.rect(self.screen, (255, 0, 0), (self.food_position[0] * BLOCK_SIZE, self.food_position[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        
        # Update the screen
        pygame.display.update()
        
    def scene_as_matrix(self):
        # Empty: 0, Snake Head: 1, Snake Body: 2, Food: 3
        matrix = np.zeros((MAP_WIDTH, MAP_HEIGHT))
        
        # Remember that the first coordinate is the y coordinate in Pygame
        # Draw the snake
        head_x, head_y = self.snake.body[0]
        matrix[head_y, head_x] = 1
            
        for block in self.snake.body[1:]:
            x, y = block
            matrix[y, x] = 2
                
        # Draw the food
        matrix[self.food_position[1], self.food_position[0]] = 3
        
        return matrix
    
    def state(self):
        matrix = self.scene_as_matrix()
        matrix_one_hot = tf.one_hot(matrix, self.elements_count)
        
        # Make a 4 value array, indicating if there is a wall or the snake body in each direction
        # If there is one of them, the value is 1, otherwise 0
        head_x, head_y = self.snake.body[0]
        prev_head_x, prev_head_y = self.snake.body[1]
        left = 1 if head_x == 0 or (head_x - 1, head_y) in self.snake.body[:-1] or (head_x - 1, head_y) == (prev_head_x, prev_head_y) else 0
        up = 1 if head_y == 0 or (head_x, head_y - 1) in self.snake.body[:-1] or (head_x, head_y - 1) == (prev_head_x, prev_head_y) else 0
        right = 1 if head_x == MAP_WIDTH - 1 or (head_x + 1, head_y) in self.snake.body[:-1] or (head_x + 1, head_y) == (prev_head_x, prev_head_y) else 0
        down = 1 if head_y == MAP_HEIGHT - 1 or (head_x, head_y + 1) in self.snake.body[:-1] or (head_x, head_y + 1) == (prev_head_x, prev_head_y) else 0
        
        return (matrix_one_hot, np.array([left, up, right, down], dtype=bool))

    def run(self):
        pygame.init()
        self.screen = pygame.display.set_mode((MAP_WIDTH * BLOCK_SIZE, MAP_HEIGHT * BLOCK_SIZE))
        movement_queue = np.array([])
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Check if the user pressed an arrow key
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        movement_queue = np.append(movement_queue, 1)
                    elif event.key == pygame.K_DOWN:
                        movement_queue = np.append(movement_queue, 3)
                    elif event.key == pygame.K_LEFT:
                        movement_queue = np.append(movement_queue, 0)
                    elif event.key == pygame.K_RIGHT:
                        movement_queue = np.append(movement_queue, 2)
                        
            if len(movement_queue) > 0:
                self.snake.change_direction(int(movement_queue[0]))
                movement_queue = np.delete(movement_queue, 0)
            
            _, _, done = self.move(self.snake.direction)
            
            if done:
                self.reset()
                
            self.draw()

            # Slow down the game
            time.sleep(0.1)

        # Quit Pygame
        pygame.quit()
        sys.exit()
        
if __name__ == '__main__':
    scene = Scene(init_randomly=True)
    scene.run()
