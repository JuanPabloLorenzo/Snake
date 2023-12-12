import pygame
import time
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from scene import Scene

scene = Scene(init_randomly=True)
scene.reset()

model_name = sys.argv[1]

model = keras.models.load_model(f"models/{model_name}_episodes.h5")

pygame.init()
scene.screen = pygame.display.set_mode((scene.width * scene.block_size, scene.height * scene.block_size))

done = True
steps = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False
        
        
    if done:
        steps = 0
        
    if steps > 50:
        scene.reset()
        steps = 0
        
    inputs = scene.scene_as_matrix()[np.newaxis]
    inputs = tf.one_hot(inputs, scene.elements_count)
        
    pred = model.predict(inputs, verbose=False)
    action = np.argmax(pred)
    
    _, _, done = scene.move(action)
    steps += 1
    scene.draw()

    # Slow down the game
    time.sleep(0.02)
