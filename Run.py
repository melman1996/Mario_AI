from nes_py.wrappers import JoypadSpace
from datetime import datetime
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
import numpy as np
import neat
import neat.reporting 
import cv2
import pickle

class Player(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        self.x = 13
        self.y = 15
        self.w = 18
        self.h = 17   
        
    def play(self):
         self.env = gym_super_mario_bros.make('SuperMarioBros-v3')
         self.env = JoypadSpace(self.env, RIGHT_ONLY)        
         self.env.reset()        
         observation, _, _, _ = self.env.step(self.env.action_space.sample()) 
         done = False        
         net = neat.nn.FeedForwardNetwork.create(bestGenome, self.config)        
         while not done:
              self.env.render()
              observation = observation[self.y * 8:self.y * 8 + self.h * 8,self.x * 8:self.x * 8 + self.w * 8]       
              cv2.imshow('main', observation)
              observation = cv2.resize(observation, (self.w, self.h))
              observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
              observation = cv2.resize(observation, (self.w, self.h))
              imgarray = np.ndarray.flatten(observation)
              imgarray = np.interp(imgarray, (0, 254), (-1, +1))
              nnOutput = net.activate(imgarray)              
              observation, reward, done, info = self.env.step(nnOutput.index(max(nnOutput)))

if __name__ == '__main__':
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config-feedforward')
    p = neat.Population(config)
    with open(r"./winners/winner 2k.pkl", "rb") as input_file:
        bestGenome = pickle.load(input_file)
    player = Player(bestGenome, config)
    player.play()

        

