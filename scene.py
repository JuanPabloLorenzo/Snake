import pygame
from snake import Snake
import random
import time
import sys
import numpy as np

MAP_WIDTH = 5
MAP_HEIGHT = 5
BLOCK_SIZE = 15

class Scene:
    def __init__(self, using_cnn=False, init_randomly=False):
        self.height = MAP_HEIGHT + 2; self.width = MAP_WIDTH + 2
        self.block_size = BLOCK_SIZE
        self.using_cnn = using_cnn
        self.init_randomly = init_randomly
        self.reset()
        self.feature_count = self.scene_as_feature_vector().shape[0]
        self.elements_count = 5 # Empty, Snake Head, Snake Body, Food, Wall
        
    def move(self):
        reward = 0
        done = False
        prev_head = self.snake.body[0]
        self.snake.move(self.food_position)
        
        # If the snake gets closer to the food, give it a reward
        if abs(self.snake.body[0][0] - self.food_position[0]) + abs(self.snake.body[0][1] - self.food_position[1]) < abs(prev_head[0] - self.food_position[0]) + abs(prev_head[1] - self.food_position[1]):
            reward += 1
        else:
            reward -= 1
        
        if self.snake.body[0] == self.food_position:
            self.new_food_position()
            reward += 10
        elif self.snake.body[0][0] < 0 or self.snake.body[0][0] >= self.width or self.snake.body[0][1] < 0 or self.snake.body[0][1] >= self.height:
            done = True
            reward -= 30
        elif self.snake.body[0] in self.snake.body[1:]:
            done = True
            reward -= 30
            
        if done:
            self.init_snake()
            self.new_food_position()
          
        if self.using_cnn:
            next_state = self.scene_as_matrix()
        else:
            next_state = self.scene_as_feature_vector()  
        
        return next_state, reward, done
    
    def init_snake(self):
        if self.init_randomly:
            head = (random.randint(0, MAP_WIDTH - 1), random.randint(0, MAP_HEIGHT - 1))
            if head[0] == 0:
                possible = [1, 2, 3]
                tail = random.choice(possible)
                possible.remove(tail)
                direction = random.choice(possible)
            elif head[0] == MAP_WIDTH - 1:
                possible = [0, 1, 3]
                tail = random.choice(possible)
                possible.remove(tail)
                direction = random.choice(possible)
            elif head[1] == 0:
                possible = [0, 2, 3]
                tail = random.choice(possible)
                possible.remove(tail)
                direction = random.choice(possible)
            elif head[1] == MAP_HEIGHT - 1:
                possible = [0, 1, 2]
                tail = random.choice(possible)
                possible.remove(tail)
                direction = random.choice(possible)
            else:
                possible = [0, 1, 2, 3]
                tail = random.choice(possible)
                possible.remove(tail)
                direction = random.choice(possible)
                
            snake_initial_body = [head]
            if tail == 0:
                snake_initial_body.append((head[0] + 1, head[1]))
            elif tail == 1:
                snake_initial_body.append((head[0], head[1] + 1))
            elif tail == 2:
                snake_initial_body.append((head[0] - 1, head[1]))
            elif tail == 3:
                snake_initial_body.append((head[0], head[1] - 1))
        else:
            snake_initial_body = [(MAP_WIDTH // 2, MAP_HEIGHT // 2), (MAP_WIDTH // 2 + 1, MAP_HEIGHT // 2)]
        
        self.snake = Snake(snake_initial_body)
            
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
        # Empty: 0, Snake Head: 1, Snake Body: 2, Food: 3, Wall: 4
        matrix = np.zeros((MAP_WIDTH, MAP_HEIGHT))
        
        # Remember that the first coordinate is the y coordinate in Pygame
        # Draw the snake
        
        head_x, head_y = self.snake.body[0]
        if 0 <= head_x < MAP_WIDTH and 0 <= head_y < MAP_HEIGHT:
            matrix[head_y, head_x] = 1
            
        for block in self.snake.body[1:]:
            x, y = block
            if 0 <= x < MAP_WIDTH and 0 <= y < MAP_HEIGHT:
                matrix[y, x] = 2
                
        # Draw the food
        matrix[self.food_position[1], self.food_position[0]] = 3
        
        # Create another matrix, the same as matrix, but with a border of walls
        matrix_with_walls = np.ones((MAP_WIDTH + 2, MAP_HEIGHT + 2)) * 4
        matrix_with_walls[1:-1, 1:-1] = matrix
        
        return matrix_with_walls
    
    def scene_as_feature_vector(self):
        food_is_up = self.food_position[1] < self.snake.body[0][1]
        food_is_down = self.food_position[1] > self.snake.body[0][1]
        food_is_left = self.food_position[0] < self.snake.body[0][0]
        food_is_right = self.food_position[0] > self.snake.body[0][0]
        wall_is_up = self.snake.body[0][1] == 0 and self.snake.direction == 1
        wall_is_down = self.snake.body[0][1] == MAP_HEIGHT - 1 and self.snake.direction == 3
        wall_is_left = self.snake.body[0][0] == 0 and self.snake.direction == 0
        wall_is_right = self.snake.body[0][0] == MAP_WIDTH - 1 and self.snake.direction == 2
        head_x, head_y = self.snake.body[0]
        body_is_up = (head_x, head_y - 1) in self.snake.body
        body_is_down = (head_x, head_y + 1) in self.snake.body
        body_is_left = (head_x - 1, head_y) in self.snake.body
        body_is_right = (head_x + 1, head_y) in self.snake.body
        
        return np.array([food_is_up, food_is_down, food_is_left, food_is_right, wall_is_up, wall_is_down, wall_is_left, wall_is_right, body_is_up, body_is_down, body_is_left, body_is_right])
        
    def run(self):
        pygame.init()
        self.screen = pygame.display.set_mode((MAP_WIDTH * BLOCK_SIZE, MAP_HEIGHT * BLOCK_SIZE))
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Check if the user pressed an arrow key
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.snake.change_direction("UP")
                    elif event.key == pygame.K_DOWN:
                        self.snake.change_direction("DOWN")
                    elif event.key == pygame.K_LEFT:
                        self.snake.change_direction("LEFT")
                    elif event.key == pygame.K_RIGHT:
                        self.snake.change_direction("RIGHT")
            
            self.move()
            self.draw()

            # Slow down the game
            time.sleep(0.1)

        # Quit Pygame
        pygame.quit()
        sys.exit()
