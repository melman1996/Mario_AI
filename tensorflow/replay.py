import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from wrappers import wrapper
from agent import Agent
from gym_super_mario_bros.actions import RIGHT_ONLY
import numpy as np
import time

if __name__ == '__main__':
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    env = wrapper(env)

    agent = Agent(env, max_memory=30000)
    

    agent.load_model(500)

    state = env.reset()
    start = time.time()
    while True:
        env.render()

        action = agent.replay(state)

        state, reward, done, info = env.step(action)

        if done or info['flag_get']:
            break
        
