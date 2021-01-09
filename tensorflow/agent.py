import tensorflow as tf

from collections import deque
import random
import numpy as np

class Agent:
    def __init__(self, env, max_memory):
        self.env = env
        self.observation_space = env.observation_space.shape
        self.action_size = env.action_space.n

        self.learning_rate = 0.00025

        self.epsilon = 1 #chance to take random action
        self.epsilon_decay = 0.99999975 
        self.epsilon_min = 0.1
        
        self.gamma = 0.90

        self.batch_size = 32

        self.memory = deque(maxlen=max_memory)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(
            32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.observation_space
        ))
        model.add(tf.keras.layers.Conv2D(
            64, (4, 4), strides=(2, 2), activation='relu'
        ))
        model.add(tf.keras.layers.Conv2D(
            64, (3, 3), strides=(1, 1), activation='relu'
        ))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
            512, activation='relu', kernel_initializer='random_uniform'
        ))
        model.add(tf.keras.layers.Dense(
            self.action_size, activation='softmax'
        ))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def run(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            state = np.reshape(state, (1, ) + self.env.observation_space.shape)
            action = np.argmax(self.model.predict(state)[0])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return action

    def remember(self, experience):
        self.memory.append(experience)

    def experience_reply(self):
        if self.batch_size > len(self.memory):
            return

        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(np.array, zip(*batch))

        target = self.model.predict(state)
        target_next = self.target_model(next_state)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(
            np.array(state),
            np.array(target),
            batch_size = self.batch_size,
            verbose = 0
        )

    def update_target_network(self):
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())