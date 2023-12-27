import pygame
import time
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from scene import Scene

scene = Scene(init_randomly=True)
scene.reset()

model = tf.keras.models.load_model("q_network_5x5.h5")

pygame.init()
scene.screen = pygame.display.set_mode((scene.width * scene.block_size, scene.height * scene.block_size))

done = True
steps = 0
total_reward = 0
sleep_time = 0.02

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False
        
    inputs = scene.state() # Inputs has 3 elements: matrix, obstacles, no_body_blocks
    
    matrix = np.expand_dims(inputs[0], axis=0)
    obstacles = np.expand_dims(inputs[1], axis=0)
    no_body_blocks = np.expand_dims(inputs[2], axis=0)
        
    pred = model.predict([matrix, obstacles, no_body_blocks], verbose=0)[0]
    action = np.argmax(pred)
    
    _, reward, done = scene.move(action)
    total_reward += reward
    
    if done or steps > 300:
        scene.reset()
        steps = 0
        print("Total reward:", total_reward)
        total_reward = 0
    else:
        steps += 1
        
    scene.draw()

    # Slow down the game
    if total_reward >= 75:
        time.sleep(0.3)
    time.sleep(sleep_time)
