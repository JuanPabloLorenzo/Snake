# Create my custom environment
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from scene import Scene
import numpy as np
import tensorflow as tf

class SnakeGame(PyEnvironment):
    def __init__(self, scene: Scene):
        super().__init__()
        self.scene = scene
        self.reset()
        
    def observation_spec(self):
        scene_spec = array_spec.BoundedArraySpec(
            shape=(self.scene.height, self.scene.width, self.scene.elements_count),
            dtype=np.uint8,
            minimum=0.0,
            maximum=1.0,
            name='scene'
        )
        
        obstacles_spec = array_spec.ArraySpec(
            shape=(4,),
            dtype=bool,
            name='obstacles'
        )
        
        no_body_spec = array_spec.ArraySpec(
            shape=(4,),
            dtype=np.float32,
            name='no_body'
        )

        return (
            scene_spec,
            obstacles_spec,
            no_body_spec
        )
        
    def action_spec(self):
        # Define the action space using ArraySpec
        return array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')

    def _reset(self):
        """Starts a new sequence and returns the first `TimeStep` of this sequence.

        Returns:
          A `TimeStep` namedtuple containing:
            step_type: A `StepType` of `FIRST`.
            reward: 0.0, indicating the reward.
            discount: 1.0, indicating the discount.
            observation: A NumPy array, or a nested dict, list, or tuple of arrays
              corresponding to `observation_spec()`.
        """
        self.scene.reset()
        observation = self.scene.state()
        
        return ts.TimeStep(
            step_type=ts.StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=observation
        )

    def _step(self, action):
        """Updates the environment according to the action and returns a `TimeStep`.

        Args:
          action: The action to be executed.

        Returns:
          A `TimeStep` namedtuple containing:
            step_type: A `StepType` of `MID`, `LAST` or `FIRST` depending on
              if the current `TimeStep` was terminal, which depends on the
              environment.
            reward: A reward of scalar value depending on the action taken.
            discount: A discount scalar between 0 and 1.
            observation: A NumPy array, or a nested dict, list or tuple of arrays
              corresponding to `observation_spec()`.
        """
        next_obs, reward, done = self.scene.move(action)
        
        return ts.TimeStep(
            step_type=ts.StepType.MID if not done else ts.StepType.LAST,
            reward=tf.constant(reward, dtype=tf.float32),
            discount=tf.constant(0.95, dtype=tf.float32),
            observation=next_obs
        )
