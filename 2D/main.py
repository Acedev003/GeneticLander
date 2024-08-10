
import pygame
import pymunk
import pymunk.pygame_util
from game import Simulation
from utils import generate_noise , pairwise

if __name__ == '__main__':
    sim = Simulation()
    sim.run()