import tensorflow as tf
from tensorflow.keras import backend as K

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
        self.epsilon_min = 0.01
        
        self.gamma = 0.90

        self.batch_size = 32

        self.memory = deque(maxlen=max_memory)
        self.burnin = 100000

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
        model.compile(loss=self._huber_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def run(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            state = np.reshape(state, (1, ) + self.env.observation_space.shape)
            action = np.argmax(self.model.predict(state)[0])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return action

    def replay(self, state):
        state = np.reshape(state, (1, ) + self.env.observation_space.shape)
        prediction = self.target_model.predict(state)[0]
        print(prediction)
        return np.argmax(prediction)

    def remember(self, experience):
        self.memory.append(experience)

    def experience_reply(self):
        if self.batch_size > len(self.memory) or len(self.memory) < self.burnin:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(np.array, zip(*batch))

        q = self.model(state).numpy()
        next_q = self.target_model(next_state).numpy()

        a = np.argmax(q, axis=1)
        target_q = reward + (1. - done) * self.gamma * next_q[np.arange(0, self.batch_size), a]
        
        self.model.fit(
            np.array(state),
            np.array(target_q),
            batch_size = self.batch_size,
            verbose = 0
        )

    def update_target_network(self):
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, episode):
        self.model.save("models/e_{}/online".format(episode))
        self.target_model.save("models/e_{}/target".format(episode))

    def load_model(self, episode, compile=False):
        self.model = tf.keras.models.load_model("models/e_{}/online".format(episode), compile=compile)
        self.target_model = tf.keras.models.load_model("models/e_{}/target".format(episode), compile=compile)