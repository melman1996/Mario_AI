from nes_py.wrappers import JoypadSpace
from datetime import datetime
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
import numpy as np
import neat
import neat.reporting 
import cv2
import pickle
class FileReporter(neat.reporting.BaseReporter):
    def __init__(self):
        with open("results.txt", "a") as file_object:  
            file_object.write("Starting Learning {0}\n".format(datetime.now()))
    def start_generation(self, generation):
        with open("results.txt", "a") as file_object:  
            file_object.write("{0}. ".format(generation))

    def post_evaluate(self, config, population, species, best_genome): 
        sum =0
        counter =0
        for key in population:
            counter = counter +1
            sum = sum + population[key].fitness
        with open("results.txt", "a") as file_object:  
            file_object.write(" best genome: {0}, Best fitness: {1}, Mean {2}\n".format(best_genome.key,best_genome.fitness,sum/counter))
        
        
    def found_solution(self,config, generation, best):
        print("found_solution")
        with open("results.txt", "a") as file_object:  
            file_object.write("Found solution")


class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        self.x = 13
        self.y = 15
        self.w = 18
        self.h = 17   
        
    def work(self):
        self.env = gym_super_mario_bros.make('SuperMarioBros-v3')
        self.env = JoypadSpace(self.env, RIGHT_ONLY)        
        self.env.reset()        
        observation, _, _, _ = self.env.step(self.env.action_space.sample())        
        # inx = int(observation.shape[0]/8)
        # iny = int(observation.shape[1]/8)
        done = False        
        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)        
        max_fitness = 0
        fitness = 0
        xpos = 0
        xpos_max = 0
        counter = 0    
        while not done:
            # self.env.render()
            observation = observation[self.y*8:self.y*8+self.h*8,self.x*8:self.x*8+self.w*8]       
            # cv2.imshow('main', observation)

            observation = cv2.resize(observation, (self.w, self.h))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = np.reshape(observation, (self.w, self.h))

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
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-1179')
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(FileReporter())
    p.add_reporter(neat.Checkpointer(10))

    pe = neat.ParallelEvaluator(4, eval_genomes)

    winner = p.run(pe.evaluate)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

