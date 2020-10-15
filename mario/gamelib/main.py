import pygame, os
from pygame.locals import *

from . import data, game

def main():
    os.environ["SDL_VIDEO_CENTERED"] = "1"
    #pygame.mixer.pre_init(44100, -16, 2, 4096)
    pygame.init()
    pygame.mouse.set_visible(0)
    pygame.display.set_icon(pygame.image.load(data.filepath("bowser1.gif")))
    pygame.display.set_caption("Super Mario Python")
    screen = pygame.display.set_mode((640, 480))
    game.Game(screen)
