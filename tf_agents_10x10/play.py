import pygame
import time
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from scene import Scene

SIZE = 10
scene = Scene(init_randomly=True, map_height=SIZE, map_width=SIZE, block_size=15)
scene.reset()

model = tf.keras.models.load_model("q_network_" + str(SIZE) + "x" + str(SIZE) + ".h5")

pygame.init()
scene.screen = pygame.display.set_mode((scene.width * scene.block_size, scene.height * scene.block_size))

done = True
steps = 0
total_reward = 0
sleep_time = 0.05

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False
        
    inputs = scene.state()
    
    matrix = np.expand_dims(inputs, axis=0)
    # obstacles = np.expand_dims(inputs[1], axis=0)
    # no_body_blocks = np.expand_dims(inputs[2], axis=0)
    # food_direction = np.expand_dims(inputs[3], axis=0)
    
    # Make the above parametric
    #inputs_expanded = [np.expand_dims(x, axis=0) for x in inputs]
        
    pred = model.predict(matrix, verbose=0)[0]
    action = np.argmax(pred)
    
    _, reward, done = scene.move(action)
    total_reward += reward
    
    if done or steps > 1000:
        scene.reset()
        steps = 0
        print("Total reward:", total_reward)
        total_reward = 0
    else:
        steps += 1
        
    scene.draw()

    # Slow down the game
    # if total_reward >= 75:
    #     time.sleep(0.3)
    time.sleep(sleep_time)
