import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from wrappers import wrapper
from agent import Agent
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import time

if __name__ == '__main__':
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = wrapper(env)

    agent = Agent(env, max_memory=100000)
    
    episodes = 10000

    starting_episode = 0
    if starting_episode > 0:
        agent.load_model(starting_episode)

    step = 0
    start = time.time()

    for e in range(starting_episode, episodes):
        total_reward = 0
        iter = 0

        state = env.reset()
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

            state = next_state

            iter += 1
            step += 1

            if done or info['flag_get']:
                break

        if e % 10 == 0:
            print('Episode {e} - '
                    'Frame {f} - '
                    'Frames/sec {fs:.0f} - '
                    'Epsilon {eps:.3f} '.format(
                        e=e,
                        f=step,
                        fs=step/(time.time()-start),
                        eps=agent.epsilon
                    )
            )
            step = 0
            start = time.time()

        if e % 100 == 0:
            agent.save_model(e)

            test_reward = 0
            max_x = 0
            state = env.reset()
            since_last_move = 0
            while True:
                action = agent.replay(state)

                state, reward, done, info = env.step(action)
                test_reward += reward

                if info['x_pos'] > max_x:
                    max_x = info['x_pos']
                else:
                    since_last_move += 1
                    if since_last_move >= 250:
                        done = True

                if done or info['flag_get']:
                    break
            print("----------Model has been saved: distance {}, total reward {}----------".format(max_x, test_reward))
        
