import tensorflow as tf

from collections import deque

class Agent:
    def __init__(self, env, max_memory):
        self.env = env
        self.observation_space = env.observation_space.shape
        self.action_size = env.action_space.n

        self.learning_rate = 0.00025

        self.epsilon = 1 #chance to take random action
        self.epsilon_decay = 0.99999975 
        self.epsilon_min = 0.1

        self.memory = deque(maxlen=max_memory)

        self.model = self.build_model()

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
        mode.add(tf.keras.layers.Dense(
            self.action_size, activation='softmax'
        ))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def run(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            action = np.argmax(self.model.predict(state)[0])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return action

    def remember(experience):
        self.memory.append(experience)