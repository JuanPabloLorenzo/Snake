import pygame
import time
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tf_agents.policies.policy_saver import PolicySaver
from snake_game import SnakeGame
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from scene import Scene

# Load the agent
saved_policy = tf.saved_model.load("policy")


# Create Scene
scene = Scene(using_cnn=True, init_randomly=True)

# Create env
env = SnakeGame(scene)
env = TFPyEnvironment(env)
time_step = env.reset()

snake_game_env = env.pyenv.envs[0]
scene = snake_game_env.scene

pygame.init()
scene.screen = pygame.display.set_mode((scene.width * scene.block_size, scene.height * scene.block_size))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False
        
    action_step = saved_policy.action(time_step)
    time_step = env.step(action_step.action)
    
    scene.draw()

    # Slow down the game
    time.sleep(0.1)
