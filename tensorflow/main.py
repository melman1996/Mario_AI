import gym_super_mario_bros
import nes_py.wrappers import JoypadSpace
from wrappers import wrapper
from agent import Agent

if __name__ == '__main__':
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, gym_super_mario_bros.actions.RIGHT_ONLY)
    env = wrapper(env)

    agent = Agent(env, max_memory=100000)
    
    episodes = 10000
    rewards = []

    for e in range(episodes):
        state = env.reset()

        total_reward = 0
        iter = 0

        while True:
            # env.render()

            action = agent.run(state)

            observation, reward, done, info = env.step(action)

            agent.remember(experience=(state, observation, action, reward, done))

            total_reward += reward

            iter += 1

            if done or info['flag_get']:
                break

        reward.append(total_reward/iter)

        if e % 100 == 0:
            print('Episode {e} - '
                    'Frame {f} - '
                    'Epsilon {eps} - '
                    'Mean reward {r}'
            )
        
