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

    agent = Agent(env, max_memory=100000)
    
    episodes = 10000
    rewards = []

    starting_episode = 0
    if starting_episode > 0:
        agent.load_model(starting_episode)

    for e in range(starting_episode, episodes):
        total_reward = 0
        iter = 0

        state = env.reset()
        start = time.time()
        while True:
            # env.render()

            action = agent.run(state)

            next_state, reward, done, info = env.step(action)

            agent.remember(experience=(state, next_state, action, reward, done))

            if iter % 4 == 0:
                agent.experience_reply()

            if iter % 1000 == 0:
                agent.update_target_network()

            total_reward += reward

            iter += 1

            if done or info['flag_get']:
                break

        rewards.append(total_reward/iter)
        print('Episode {e} - '
                'Frame {f} - '
                'Frames/sec {fs} - '
                'Epsilon {eps} - '
                'Mean reward {r}'.format(
                    e=e,
                    f=iter,
                    fs=iter/(time.time()-start),
                    eps=agent.epsilon,
                    r=total_reward
                )
        )
        if e % 100 == 0:
            agent.save_model(e)
        
