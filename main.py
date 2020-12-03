from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import neat
import cv2
import pickle

class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        
    def work(self):
        self.env = gym_super_mario_bros.make('SuperMarioBros-v0')
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        
        self.env.reset()
        
        observation, _, _, _ = self.env.step(self.env.action_space.sample())
        
        inx = int(observation.shape[0]/8)
        iny = int(observation.shape[1]/8)
        done = False
        
        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        
        max_fitness = 0
        fitness = 0
        xpos = 0
        xpos_max = 0
        counter = 0

        # cv2.namedWindow('main', cv2.WINDOW_NORMAL)
        
        while not done:
            # self.env.render()

            # scaledimg = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
            # scaledimg = cv2.resize(scaledimg, (iny, inx))
            # cv2.imshow('main', scaledimg)
            # cv2.waitKey(1)

            observation = cv2.resize(observation, (inx, iny))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = np.reshape(observation, (inx, iny))

            imgarray = np.ndarray.flatten(observation)
            imgarray = np.interp(imgarray, (0, 254), (-1, +1))
            nnOutput = net.activate(imgarray)

            observation, reward, done, info = self.env.step(nnOutput.index(max(nnOutput)))

            fitness += int(reward)

            if fitness > max_fitness:
                max_fitness = fitness
                counter = 0
            else:
                counter += 1
                
            if done or counter > 350 or info['life'] < 2:
                done = True
                fitness += info['score']
                
            if info['flag_get']:
                fitness += 100000

        return fitness


def eval_genomes(genome, config):
    worky = Worker(genome, config)
    return worky.work()

if __name__ == '__main__':
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config-feedforward')

    p = neat.Population(config)
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-629')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))

    pe = neat.ParallelEvaluator(10, eval_genomes)

    winner = p.run(pe.evaluate)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

