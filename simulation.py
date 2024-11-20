import os
import csv
import neat
import time
import pickle
import random
import neat.genome
import pygame
import pygame.gfxdraw
import pymunk
import datetime
import configparser

from pymunk.pygame_util import DrawOptions

from lander import TwinFlameCan
from utils import plot_stats,plot_species,pairwise,Noise

class GeneticSimulation:
    def __init__(self,
                 simulation_config_file : str,
                 lander_config_file  : str,
                 terrain_config_file : str,
                 headless       : bool = False):
        
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy" 
            
        self.headless   = headless

        self.simulation_config = configparser.ConfigParser()
        self.lander_config     = configparser.ConfigParser()
        self.terrain_config    = configparser.ConfigParser()

        self.simulation_config.read(simulation_config_file)
        self.lander_config.read(lander_config_file)
        self.terrain_config.read(terrain_config_file)

        self.simulation_config_file = simulation_config_file
        self.run_folder  = f'runs/{datetime.datetime.now()}'

        self.generations = int(self.simulation_config['SIMULATION']['GENERATIONS'])

        self.sim_width    = int(self.simulation_config['SIMULATION']['SIM_WIDTH'])
        self.stat_width   = int(self.simulation_config['SIMULATION']['STAT_WIDTH'])
        self.screen_width = self.sim_width + self.stat_width

        self.height     = int(self.simulation_config['SIMULATION']['HEIGHT'])

        pygame.init()
        pygame.display.set_caption('Genetic Lander')

        self.render_screen = pygame.display.set_mode((self.screen_width, self.height))
        self.sim_screen    = self.render_screen.subsurface((0, 0, self.sim_width, self.height))
        self.stat_screen   = self.render_screen.subsurface((self.sim_width,0,200,self.height))

        self.clock = pygame.time.Clock()
        self.fps   = int(self.simulation_config['SIMULATION']['FPS'])

        self.terrain = {
            "texture" : pygame.image.load(self.terrain_config['CONFIG']['texture']).convert(),
            "body": None,
            "segments" : [],
            "segment_coords" : [],
            "segment_length" : int(self.terrain_config['CONFIG']['segment_length'])
        }

        self.gravity        = float(self.simulation_config['SIMULATION']['GRAVITY'])
        self.space          = pymunk.Space()
        self.space.gravity  = (0, self.gravity)

        self.category = {
            "terrain": 0b01,
            "lander":  0b10
        } 

        self.mask = {
            "terrain": 0b10,
            "lander":  0b01
        }

        self.landers:list[TwinFlameCan]  = []
        self.focused_lander:TwinFlameCan = None

        self.collion_handler = self.space.add_collision_handler(self.category['lander'],self.category['terrain'])
        self.collion_handler.post_solve = self.handle_collision

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            elif event.type ==pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_ESCAPE:
                    exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_click(event)

    def handle_mouse_click(self,event):
        pos = event.pos
        for lander in self.landers:
            if lander.shape.point_query(pos).distance < 0:
                self.focused_lander = lander
                return
    
    def handle_collision(self):
        return

    def display_stat(self,paused):
        if not self.focused_lander:
            return
        
        lander_texture = self.focused_lander.lander_texture
        screen_width   = self.stat_width
        texture_width  = lander_texture.get_width()

        self.stat_screen.blit(lander_texture,(screen_width/2 - texture_width/2,10))

        font = pygame.font.SysFont(None, 18)

        img = font.render(f"VEL X: {self.focused_lander.vel_x:.3f}", False, "WHITE")
        self.stat_screen.blit(img, (5, 80))

        img = font.render(f"VEL Y: {self.focused_lander.vel_y:.3f}", False, "WHITE")
        self.stat_screen.blit(img, (5, 100))

        img = font.render(f"ANGLE: {self.focused_lander.angle:.3f}", False, "WHITE")
        self.stat_screen.blit(img, (5, 120))

        img = font.render(f"R VEL: {self.focused_lander.angular_vel:.3f}", False, "WHITE")
        self.stat_screen.blit(img, (5, 140))

        img = font.render(f"SLOPE: {self.focused_lander.slope:.3f}", False, "WHITE")
        self.stat_screen.blit(img, (5, 160))

        img = font.render(f"HGHT : {self.focused_lander.distance_to_surface:.3f}", False, "WHITE")
        self.stat_screen.blit(img, (5, 180))

        img = font.render(f"LAND?: {self.focused_lander.landed}", False, "WHITE")
        self.stat_screen.blit(img, (5, 200))

        img = font.render(f"ALIVE: {self.focused_lander.alive}", False, "WHITE")
        self.stat_screen.blit(img, (5, 220))

        img = font.render(f"COD  : {self.focused_lander.cause_of_death}", False, "WHITE")
        self.stat_screen.blit(img, (5, 240))

        img = font.render(f"ID   : {self.focused_lander.body.id}", False, "WHITE")
        self.stat_screen.blit(img, (5, 260))

        img = font.render(f"IMP  : {self.focused_lander.impulse}", False, "WHITE")
        self.stat_screen.blit(img, (5, 280))

        if paused:
            pygame.display.flip()

    def generate_terrain(self):
        noise_generator     = Noise()
        base_surface_height = int(self.terrain_config['CONFIG']['base_surface_height'])
        min_surface_height  = int(self.terrain_config['CONFIG']['min_surface_height'])

        self.base_height    = self.height - base_surface_height

        prev_x = 0
        prev_y = self.base_height

        for x in range(self.terrain["segment_length"], self.sim_width, self.terrain["segment_length"]):
        
            noise_value = noise_generator.generate_noise([x / self.sim_width, 0]) * 100
            y = self.base_height + noise_value
            y = min(self.height - min_surface_height, y)
            self.terrain["segment_coords"].append(((prev_x, prev_y), (x, y)))
            prev_x = x
            prev_y = y

        if prev_x < self.sim_width:
            noise_value = noise_generator.generate_noise([prev_x / self.sim_width, 0]) * 100
            y = self.base_height + noise_value
            y = min(self.height - min_surface_height, y)
            self.terrain["segment_coords"].append(((prev_x, prev_y), (self.sim_width, y)))

        # Create the static body for the terrain
        self.terrain["body"] = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.terrain["body"].position = self.terrain["segment_coords"][0][0]

        # Create terrain segments as pymunk segments
        self.terrain["segments"] = [
            pymunk.Segment(
                self.terrain["body"], 
                (start[0], -self.base_height + start[1]+20), 
                (end[0], -self.base_height + end[1]+20), 
                20
            )
            for start, end in self.terrain["segment_coords"]
        ]

        # Add the body and segments to the pymunk space
        self.space.add(self.terrain['body'])
        for segment in self.terrain["segments"]:
            segment.friction = 1
            segment.collision_type = 1
            segment.filter = pymunk.ShapeFilter(categories=self.category['terrain'], mask=self.mask['terrain'])
            self.space.add(segment)

    def draw_terrain(self):
        for x in self.terrain["segment_coords"]:
            pygame.draw.circle(self.sim_screen,'red',x[0],4)
            pygame.draw.circle(self.sim_screen,'green',x[1],4)
        
        points = [(0,self.height),self.terrain["segment_coords"][0][0]]
        for segment in self.terrain["segment_coords"]:
            points.append(segment[1])
        points.append((self.sim_width,self.height))

        pygame.gfxdraw.textured_polygon(self.sim_screen,points,self.terrain["texture"],0,0)

    def remove_landers(self):
        for lander in self.landers:
            try:
                self.space.remove(lander.shape)
                self.space.remove(lander.body)
            except:
                pass

    def remove_terrain(self):
        for x in self.terrain["segments"]:
            self.space.remove(x)
        self.space.remove(self.terrain["body"])

        self.terrain['segments'] = []
        self.terrain['segment_coords'] = []

    def run(self,resume_path:str = None):
        os.mkdir(self.run_folder)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             self.simulation_config_file)
        
        if resume_path:
            population = neat.Checkpointer().restore_checkpoint(resume_path)
        else:
            population = neat.Population(config)

        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.Checkpointer(10,filename_prefix=f"{self.run_folder}/ckpt-"))

        winner = population.run(self.simulation, self.generations)
        pickle.dump(winner, open(os.path.join(self.run_folder, 'winner.pkl'), 'wb'))     

    def simulation(self,genomes: list[tuple[int,neat.genome.DefaultGenome]],config):
        self.landers = []

        for genome_id, genome in genomes:
            genome.fitness = 10000000
            lander = TwinFlameCan(self.sim_screen,
                                   self.space,
                                   self.terrain,
                                   {"id":genome_id,"network":neat.nn.FeedForwardNetwork.create(genome,config)},
                                   self.lander_config)
            lander.shape.filter = pymunk.ShapeFilter(categories = self.category['lander'], mask=self.mask['lander'])
            self.landers.append(lander)

        draw_options = DrawOptions(self.sim_screen)

        self.generate_terrain()

        self.running = True
        self.paused  = False
        while self.running:
            self.running = any(lander.alive and not lander.landed for lander in self.landers)
            self.handle_events()
            self.stat_screen.fill('BLACK')
            self.display_stat(self.paused)
            
            if self.paused:
                continue
            
            self.sim_screen.fill('BLACK')
            self.space.debug_draw(draw_options)
            
            self.draw_terrain()
            for lander in self.landers:
               if lander.alive:
                   lander.update()
                   lander.draw()
            
            pygame.display.flip()
            self.space.step(1/(self.fps))
            self.clock.tick(self.fps)

        self.remove_terrain()
        self.remove_landers()

