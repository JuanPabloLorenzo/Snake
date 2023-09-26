import pygame
import time
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from scene import Scene

using_cnn = sys.argv[2] == "cnn" if len(sys.argv) > 2 else False

scene = Scene(using_cnn=using_cnn, init_randomly=True)
scene.reset()

model_name = sys.argv[1]

if using_cnn:
    model = keras.models.load_model(f"models/CNN_model{model_name}.h5")
else:
    model = keras.models.load_model(f"models/model{model_name}.h5")

pygame.init()
scene.screen = pygame.display.set_mode((scene.width * scene.block_size, scene.height * scene.block_size))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False
        
    inputs = scene.scene_as_matrix()[np.newaxis] if scene.using_cnn else scene.scene_as_feature_vector()[np.newaxis]
    if using_cnn:
        inputs = tf.one_hot(inputs, scene.elements_count)
        
    pred = model.predict(inputs, verbose=False)
    action = np.argmax(pred)
    scene.snake.change_direction(action)
    
    scene.move()
    scene.draw()

    # Slow down the game
    time.sleep(0.1)
