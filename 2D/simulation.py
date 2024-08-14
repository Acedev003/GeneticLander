import csv
import neat
import time
import random
import pygame
import pymunk

from lander import Categories, Lander
from utils import plot_stats,plot_species,pairwise,Noise

class GeneticSimulation:
    def __init__(self,
                 generations   : int = 5000,
                 screen_width  : int = 1280,
                 screen_height : int = 720
                 ):
        
        pygame.init()
        pygame.display.set_caption('Planetary Lander Evolution')
            
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.width  = screen_width
        self.height = screen_height
            
        self.fps     = 24                               # Lower FPS boosts performance but may cause jitter
        
        self.dt      = 1/self.fps
        self.clock   = pygame.time.Clock()
        
        self.terrain_points       = []
        self.terrain_break_count  = 50                  # Number of polygons to make terrain
        self.terrain_complexity   = 300                 # Perlin noise param. Higher gives steeper variations
        self.min_terrain_altitude = 10                  # Lowest height of generated terrain
        self.terrain_screen_prcnt = 0.8                 # 0.5 to 0.8 recommended. Terrain height base as a percentage of screen.
        self.terrain_friction     = 0.9
        
        self.gravity        = 3.6742                    # Acceleration due to gravity
        
        self.space          = pymunk.Space()
        self.space.gravity  = (0, self.gravity*100)
        
        self.landers           = []
        self.lander_spawn_y    = 100                    # 100 to 500 recommended. Spawns lander at this y-coordinate
        self.no_spawn_margin_x = 500                    # Prevents any lander spawning in +- of this range
    
        self.neat_config_path = "neat_config.ini"       # Path to neat configuration file 
        self.fitness_file     = f'runs/fitness_data_run-{int(time.time())}.csv'
        self.generation_count = generations
        
        
        self.collion_handler = self.space.add_collision_handler(Categories.LANDER_CAT,Categories.TERRAIN_CAT)
        self.collion_handler.post_solve = self.handle_collision
     
    def run(self):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             self.neat_config_path)
        
        population = neat.Population(config)
        
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.Checkpointer(10,filename_prefix="checkpoints/ckpt-"))
        
        winner = population.run(self.run_simulation, self.generation_count)
        
        plot_stats(stats, ylog=False, view=True)
        plot_species(stats, view=True)     
    
    def run_simulation(self,genomes,config):
        self.terrain_points = []
        self.landers        = []
        
        self.generate_terrain_points()
        self.initialize_terrain_physics()
        
        self.landing_zone = self.find_landing_zone()
        
        for genome_id, genome in genomes:
            genome.fitness = 0
            
            x_min    = 50
            x_max    = self.width - 50
            target_x = self.landing_zone[0]

            x = random.randint(x_min, x_max)
            while target_x - self.no_spawn_margin_x < x < target_x + self.no_spawn_margin_x:
                x = random.randint(x_min, x_max)
                
            y = self.lander_spawn_y
            
            self.landers.append(
                Lander((x,y),
                       self.screen,
                       self.space,
                       genome_id,
                       neat.nn.FeedForwardNetwork.create(genome,config),
                       genome,
                       self.landing_zone,
                       self.terrain_points)
            )
            
        print("LANDERS_COUNT:",len(self.landers))
        
        running = True
        paused  = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                elif event.type ==pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused

            if paused:
                continue
            
            self.screen.fill('BLACK')
            self.draw_terrain()
            
            pygame.draw.circle(self.screen, (255,255,0), self.landing_zone, 5)
            
            for lander in self.landers:
                lander.update()
                lander.draw()
            
            pygame.display.flip()
            self.space.step(self.dt)        
            self.clock.tick(self.fps)
            
            running = any(lander.has_life() and not lander.get_collision_status() for lander in self.landers)

        self.remove_terrain()

        dist_sum, vel_sum, fit_sum = 0 , 0 , 0
        for lander in self.landers:
            dist,vel,fit = lander.evaluate_lander()
                
            dist_sum += dist
            vel_sum  += vel
            fit_sum  += fit
        
        num_landers  = len(self.landers)
        avg_distance = dist_sum / num_landers
        avg_velocity = vel_sum / num_landers
        avg_fitness  = fit_sum / num_landers
        
        with open(self.fitness_file, 'a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['Avg Dist','Avg Speed','Avg Fitness'])
            writer.writerow([avg_distance, avg_velocity, avg_fitness])
            
        
    def generate_terrain_points(self):
        noise_func = Noise()
        terrain_break_heights = [ noise_func.generate_noise([x/self.terrain_break_count,0]) 
                                 for x in range(self.terrain_break_count) ]
        
        start_index = 0
        points_gap  = self.width / (self.terrain_break_count - 1) 
        
        self.terrain_points.append([start_index,self.height])
        
        for index, height in enumerate(terrain_break_heights[:-1]):
            height1 = min(self.height-self.min_terrain_altitude,
                          (self.height * self.terrain_screen_prcnt) + (height * self.terrain_complexity))
            p1 = [start_index, height1]

            start_index += points_gap

            height2 = min(self.height-self.min_terrain_altitude,
                          (self.height * self.terrain_screen_prcnt) + (terrain_break_heights[index + 1] * self.terrain_complexity))
            p2 = [start_index, height2]
            
            self.terrain_points.append(p1)
            self.terrain_points.append(p2)
        
        if len(terrain_break_heights) > 1:
            final_height = min(self.height-self.min_terrain_altitude,
                               (self.height * self.terrain_screen_prcnt) + (terrain_break_heights[-1] * self.terrain_complexity))
            self.terrain_points.append([self.width,final_height])
            
        # Final Corner to close polygon
        self.terrain_points.append([self.width,self.height])       
    
    def initialize_terrain_physics(self):
        self.terrain_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.space.add(self.terrain_body)
        
        for a,b in pairwise(self.terrain_points[1:-1]):
            v1 = a
            v2 = b
            v3 = [ b[0] , self.height ]
            v4 = [ a[0] , self.height ]
            
            shape = pymunk.Poly(self.terrain_body, [v1,v2,v3,v4])
            shape.filter = pymunk.ShapeFilter(categories=Categories.TERRAIN_CAT,mask=Categories.LANDER_CAT)
            shape.collision_type = Categories.TERRAIN_CAT
            shape.friction = self.terrain_friction 
            self.space.add(shape)
            
    def find_landing_zone(self,flat_segment_width=100):
        vertices         = self.terrain_points
        best_start_index = 0
        lowest_slope_sum = float('inf')

        for start_index in range(len(vertices) - 1):
            slope_sum     = 0
            segment_width = 0
            end_index = start_index

            while end_index < len(vertices) - 1 and segment_width < flat_segment_width:
                x1, y1 = vertices[end_index]
                x2, y2 = vertices[end_index + 1]

                if x2 == x1:
                    slope = 0  # Flat segment (vertical in x)
                else:
                    slope = abs((y2 - y1) / (x2 - x1))
                slope_sum += slope
                segment_width += (x2 - x1)
                end_index += 1

            if segment_width >= flat_segment_width and slope_sum < lowest_slope_sum:
                lowest_slope_sum = slope_sum
                best_start_index = start_index

        # Calculate the center point of the flattest region
        flat_region_start_x = vertices[best_start_index][0]
        segment_width = 0
        end_index = best_start_index

        while end_index < len(vertices) - 1 and segment_width < flat_segment_width:
            segment_width += (vertices[end_index + 1][0] - vertices[end_index][0])
            end_index += 1

        flat_region_end_x = vertices[end_index][0]
        center_x = (flat_region_start_x + flat_region_end_x) / 2

        # Find the y-coordinate corresponding to the center x-coordinate
        for k in range(len(vertices) - 1):
            x1, y1 = vertices[k]
            x2, y2 = vertices[k + 1]

            if x1 <= center_x <= x2:
                t = (center_x - x1) / (x2 - x1)
                center_y = y1 + t * (y2 - y1)
                return (center_x, center_y)

        x_coords, y_coords = zip(*vertices)
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
    
        random_x = random.uniform(min_x, max_x)
        random_y = random.uniform(min_y, max_y)
    
        return (random_x, random_y)

    def remove_terrain(self):
        for shape in self.terrain_body.shapes:
            self.space.remove(shape)
        self.space.remove(self.terrain_body)
    
    def draw_terrain(self):
        if not self.terrain_points:
            return
        
        pygame.draw.polygon(self.screen, (255, 255, 255), self.terrain_points)
    
    def handle_collision(self,arbiter,space,data):
        shapes = arbiter.shapes
        
        if arbiter.is_first_contact:
            ids    = [] 
            for shape in shapes:
                ids.append(shape.body.id)
            
            for lander in self.landers:
                if lander.get_body_id() in ids:
                    lander.set_collided()
                
            
    