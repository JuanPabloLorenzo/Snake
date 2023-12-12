import numpy as np
from scene import Scene
import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import multiprocessing
import sys

def training_step(batch_size, replay_buffer, discount_rate):
    if len(replay_buffer) < batch_size * 3:
        return
    
    states, actions, rewards, next_states, dones = sample_experiences(batch_size, replay_buffer)
    next_Q_values = target.predict(next_states, verbose=False)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards + (1 - dones) * discount_rate * max_next_Q_values).reshape(-1, 1)
    mask = tf.one_hot(actions, 4)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

### Executed in the workers ###
def epsilon_greedy_policy(model, state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(4)
    else:
        Q_values = model.predict(state[np.newaxis], verbose=0)
        return np.argmax(Q_values[0])


def sample_experiences(batch_size, replay_buffer):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = []
    for index in indices:
        try:
            batch.append(replay_buffer[index]) # The queue.qsize() is not reliable
        except:
            pass
            
    states, actions, rewards, next_states, dones = [np.array([experience[field_index] for experience in batch])
                                                            for field_index in range(5)]
    return states, actions, rewards, next_states, dones

def play_one_step(model, scene, state, epsilon, replay_buffer):
    action = epsilon_greedy_policy(model, state, epsilon)
    next_state, reward, done = scene.move(action)
    next_state = tf.one_hot(next_state, scene.elements_count)
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done

def experience_collector(model_weigths, scene, epsilon, episodes, avg_reward_queue, replay_buffer):
    model = keras.Sequential([
        Conv2D(12, (3, 3), activation='relu', padding="same", kernel_initializer='he_normal', input_shape=(scene.height, scene.width, scene.elements_count)),
        Conv2D(12, (3, 3), activation='relu', strides=2, padding="same", kernel_initializer='he_normal'),
        Flatten(),    
        Dense(16, activation='relu', kernel_initializer='he_normal'),
        Dense(4, activation='linear')
    ])
    model.set_weights(model_weigths)
    
    avg_reward = 0
    for episode in range(1, episodes + 1):
        scene.reset()
        state = scene.scene_as_matrix()
        state = tf.one_hot(state, scene.elements_count)
        done = False
        steps = 0
        
        while not done and steps < 200:
            steps += 1
            _, reward, done = play_one_step(model, scene, state, epsilon, replay_buffer)
            avg_reward += reward

    avg_reward /= episodes
    avg_reward_queue.put(avg_reward)


if __name__ == '__main__':
    scene = Scene(init_randomly=True)

    model = keras.Sequential([
        Conv2D(12, (3, 3), activation='relu', padding="same", kernel_initializer='he_normal', input_shape=(scene.height, scene.width, scene.elements_count)),
        Conv2D(12, (3, 3), activation='relu', strides=2, padding="same", kernel_initializer='he_normal'),
        Flatten(),    
        Dense(16, activation='relu', kernel_initializer='he_normal'),
        Dense(4, activation='linear')
    ])


    optimizer = keras.optimizers.legacy.Adam(learning_rate=1e-3)
    loss_fn = keras.losses.mean_squared_error
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    target = keras.models.clone_model(model)
    target.set_weights(model.get_weights())
    
    batch_size = 32
    discount_rate = 0.95

    # Parallel trainig: many processes for generating experiences and one process for training
    # The experience generation and the training are not done in parallel
    # The experience generation is done in parallel and is the bottleneck
    workers_count = 8
    episodes_count = 2400
    epochs = 20
    episodes_per_epoch = episodes_count // epochs
    episodes_per_worker_per_epoch = episodes_per_epoch // workers_count
    episodes_per_workes_first_epoch = episodes_per_worker_per_epoch*5
    
    # Queue for the workers to save the average reward
    avg_reward_queue = multiprocessing.Queue()
    # (state, action, reward, next_state, done)
    # replay_buffer = multiprocessing.Queue()
    replay_buffer = multiprocessing.Manager().list()
    
    for epoch in range(epochs):
        print("Epoch:", epoch)
        processes = []
        epsilon = 1.0 - epoch / epochs
        model_weights = model.get_weights()
        
        worker_episodes = episodes_per_worker_per_epoch if epoch > 0 else episodes_per_workes_first_epoch
        for i in range(workers_count):
            process = multiprocessing.Process(
                target=experience_collector,
                args=(model_weights, scene, epsilon, worker_episodes, avg_reward_queue, replay_buffer)
            )
            processes.append(process)
            process.start()
            
        # Join all processes
        for process in processes:
            process.join()
            
        # Get the average reward and print it
        avg_reward = 0
        for i in range(workers_count):
            avg_reward += avg_reward_queue.get()
            
        avg_reward /= workers_count
        print("Average reward:", avg_reward)

        # Update the target model
        if (epoch + 1) % 2 == 0:
            target.set_weights(model.get_weights())        
            
        # Train the model
        for step in range(episodes_per_epoch):
            training_step(batch_size, replay_buffer, discount_rate)
            
        # Save the model
        if (epoch + 1) % 5 == 0:
            model.save(f"models/{epoch*episodes_per_epoch}_episodes.h5")
          
        
                
