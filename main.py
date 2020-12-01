from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import neat
import cv2
import pickle

def eval_genomes(genomes, config):
    for i, genome, in genomes:
        observation = env.reset()
        action = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape
        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0

        done = False

        while not done:
            env.render()
            frame += 1

            observation = cv2.resize(observation, (inx, iny))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = np.reshape(observation, (inx, iny))

            imgarray = observation.flatten()
            nnOutput = net.activate(imgarray)

            observation, reward, done, info = env.step(nnOutput.index(max(nnOutput)))

            fitness_current = info['x_pos'] + info['score']
            #fitness_current += reward

            if info['flag_get']:
                fitness_current += 100000

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if counter > 250: #kill mario if he is not making progress
                done = True
            
            genome.fitness = fitness_current



env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)# RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

config =  neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedworward')
p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
