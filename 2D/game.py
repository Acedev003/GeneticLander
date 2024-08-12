import math
import time
import neat
import pymunk
import pygame
import random
import visualize
from utils import generate_noise , pairwise , Noise

class Categories:
    LANDER_CAT  = 0b01
    TERRAIN_CAT = 0b10
    
class SmokeParticle:
    def __init__(self, position, velocity, life_time, screen):
        self.position = position
        self.velocity = velocity
        self.life_time = life_time
        self.initial_life_time = life_time
        self.screen = screen

    def update(self, dt):
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt
        self.life_time -= dt

    def draw(self):
        if self.life_time > 0:
            alpha = max(0, int(255 * (self.life_time / self.initial_life_time)))
            color = (255, 255, 255, alpha)
            surface = pygame.Surface((5, 5), pygame.SRCALPHA)
            pygame.draw.circle(surface, color, (2, 2), 2)
            self.screen.blit(surface, self.position)

    def is_alive(self):
        return self.life_time > 0
    
class SmokeEmitter:
    def __init__(self, screen):
        self.particles = []
        self.screen = screen

    def emit(self, position):
        velocity = [random.uniform(-1, 1), random.uniform(-2, 0)]
        life_time = 0.2#random.uniform(0.5, 1.0)
        self.particles.append(SmokeParticle(list(position), velocity, life_time, self.screen))

    def update_and_draw(self, dt):
        for particle in self.particles:
            particle.update(dt)
            particle.draw()

        # Remove dead particles
        self.particles = [p for p in self.particles if p.is_alive()]

class Lander:
    def __init__(self,position,screen,space,mass,id=None,network=None,genome=None,target_zone=None):
        
        self.screen   = screen
        self.screen_w = screen.get_size()[0]
        self.screen_h = screen.get_size()[1]
        self.space    = space
        
        self.image = pygame.image.load("assets/Lander.png")
        self.image = self.image.convert_alpha()
        
        self.body = pymunk.Body()
        self.body.position = position
        self.genome_id = id
        self.body_id   = self.body.id
        self.network = network
        self.genome  = genome 
        self.target_zone = target_zone
        
        
        self.altimer_scan_length = 100
        
        space.add(self.body)
        
        self.smoke_emitter = SmokeEmitter(screen)
        
        self.segments = [
            ([15,0],[35,0]),
            ([35,0],[44,10]),
            ([44,10],[44,38]),
            ([44,38],[49,49]),
            ([44,38],[5,38]),
            ([5,38],[1,49]),
            ([5,38],[5,10]),
            ([5,10],[15,0])
        ]
        
        for a,b in self.segments:
            segment = pymunk.Segment(self.body, a, b, 2)
            segment.color = (255,0,0,0)
            segment.mass = mass/(len(self.segments)+1)
            segment.filter = pymunk.ShapeFilter(categories=Categories.LANDER_CAT,mask=Categories.TERRAIN_CAT)
            segment.collision_type = Categories.LANDER_CAT
            segment.friction = 1
            segment.elasticity = 0
            space.add(segment)
        
        center_span_a = [5,25]
        center_span_b = [44,25]
        
        center_span = pymunk.Segment(self.body, center_span_a, center_span_b, 2)
        center_span.color = (255,0,0,0)
        center_span.mass = mass/(len(self.segments)+1)
        center_span.filter = pymunk.ShapeFilter(categories=Categories.LANDER_CAT,mask=Categories.TERRAIN_CAT)
        center_span.collision_type = Categories.LANDER_CAT
        space.add(center_span)
        
        self.center_span = center_span
        
        ## Input Parameters
        
        self.roll_percentage  = 1.0
        self.velocity         = 0
        self.x_pos            = self.center_span.bb.center()[0]
        self.y_pos            = self.center_span.bb.center()[1]
        self.x_pos_percentage = self.x_pos/self.screen_w
        self.y_pos_percentage = self.y_pos/self.screen_h
        self.left_alt_probe   = 0
        self.right_alt_probe  = 0
        
        self.alive              = True
        self.visible            = True
        self.has_collided       = False
        self.collision_velocity = 10000000
        
    def draw_and_update(self):
        
        angle_degrees = math.degrees(self.body.angle)
        rotated_image = pygame.transform.rotate(self.image, -angle_degrees)
        
        center_x,center_y = self.center_span.bb.center()
        
        center_span_angle          = self.center_span.body.angle
        center_span_angle_deg      = math.degrees(center_span_angle)
        center_span_angle_deg_norm = center_span_angle_deg % 360
        
        if center_span_angle_deg_norm > 270 or center_span_angle_deg_norm < 90:
            force = -10000 * int(not self.has_collided)
            fx    = force * math.sin(-self.center_span.body.angle)
            fy    = force * math.cos(-self.center_span.body.angle)
        else:
            force = 10000 * int(not self.has_collided)
            fx    = force * math.sin(-self.center_span.body.angle)
            fy    = force * math.cos(-self.center_span.body.angle)
        
        alt_l , alt_r         = self.get_altimeter_readings()  
        
        self.roll_percentage  = min(center_span_angle_deg_norm,360-center_span_angle_deg_norm)/360
        self.velocity         = abs(self.body.velocity)
        self.x_pos            = self.center_span.bb.center()[0]
        self.y_pos            = self.center_span.bb.center()[1]
        self.x_pos_percentage = self.x_pos/self.screen_w
        self.y_pos_percentage = self.y_pos/self.screen_h
        self.left_alt_probe   = alt_l
        self.right_alt_probe  = alt_r
        
        x1, y1 = self.x_pos , self.y_pos
        x2, y2 = self.target_zone
        
        self.distance_to_trgt = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        self.to_left_right    = x2-x1
        
        # print("ID:",self.genome_id)
        # #print("CSPAN_ANG_RAD:",center_span_angle)
        # #print("CSPAN_ANG_DEG:",center_span_angle_deg)
        # print("CSPAN_ANG_DEG_NORM:",center_span_angle_deg_norm)
        # print("ROLL_PERCENT:",self.roll_percentage)
        # print("VELOCITY:",self.velocity)
        # print("POS_X:",self.x_pos)
        # print("POS_Y:",self.y_pos)
        # print('ALT_L:',self.left_alt_probe)
        # print('ALT_R:',self.right_alt_probe)
        # #print("POS_X_PERCENT:",self.x_pos_percentage)
        # #print("POS_Y_PERCENT:",self.y_pos_percentage)
        # print("FORCE:",force)
        # #print("FX",fx)
        # #print("FY",fy)
        # print("HAS_COLLIDED:",self.has_collided)
        # print("COLLISION_VEL:",self.collision_velocity)
        # print("#######")

        engine_data    = self.network.activate([self.left_alt_probe,
                                                self.right_alt_probe,
                                                self.to_left_right,
                                                self.velocity,
                                                center_span_angle_deg_norm])
        engine_force_l = engine_data[0]
        engine_force_r = engine_data[1]

        
        fxl = engine_force_l * fx
        fyl = engine_force_l * fy
        
        fxr = engine_force_r * fx
        fyr = engine_force_r * fy
        
        self.body.apply_force_at_local_point((fxl,fyl), self.center_span.a)
        self.body.apply_force_at_local_point((fxr,fyr), self.center_span.b)
        
        if engine_force_l > 0.5:
            self.smoke_emitter.emit(self.body.local_to_world(self.center_span.a))
        if engine_force_r > 0.5:
            self.smoke_emitter.emit(self.body.local_to_world(self.center_span.b))
        
        rotated_rect  = rotated_image.get_rect(center=(center_x,center_y))
        
        if self.visible and self.alive:
            self.screen.blit(rotated_image,rotated_rect)
            self.smoke_emitter.update_and_draw(1/60)

        if self.body.position[0] > self.screen_w or self.body.position[0] < 0 or self.body.position[1] > self.screen_h or self.body.position[1] < 0:
            self.kill()
    
    def set_collision_data(self):
        if self.has_collided:
            return
        
        self.has_collided = True
        self.collision_velocity = self.velocity
        if self.collision_velocity > 850:
            self.kill()
    
    
    def kill(self):
        for shape in self.body.shapes:
            self.space.remove(shape)
        self.space.remove(self.body)
        self.alive = False
        
    def has_life(self):
        return self.alive
    
    def get_collision_status(self):
        return self.has_collided
    
    def get_genome_id(self):
        return self.genome_id
    
    def get_body_id(self):
        return self.body_id
    
    
    def get_altimeter_readings(self):
        
        coord_left  = (self.x_pos - self.altimer_scan_length//2,self.y_pos)
        coord_right = (self.x_pos + self.altimer_scan_length//2,self.y_pos)
        
        info_left = self.space.point_query_nearest(coord_left, self.screen_h,shape_filter = pymunk.ShapeFilter(categories=Categories.LANDER_CAT,mask=Categories.TERRAIN_CAT))
        info_right = self.space.point_query_nearest(coord_right,self.screen_h,shape_filter = pymunk.ShapeFilter(categories=Categories.LANDER_CAT,mask=Categories.TERRAIN_CAT))
        
        return info_left.distance, info_right.distance

    def evaluate_fitness(self):
        
        roll_score   = (1-self.roll_percentage) * 100
        
        if self.collision_velocity > 850:
            speed_score = 0
        else:
            speed_score = 850-self.collision_velocity
        
        speed_score = speed_score**4
        
        life_bonus   = float(self.alive) 
        
        dist_score   = self.distance_to_trgt ** 2
        
        total_fitness =  (roll_score + speed_score - dist_score ) * life_bonus
        
        self.genome.fitness = total_fitness

class LanderFactory:
    def __init__(self,screen,space,num_landers=50):
        self.screen = screen
        self.space  = space
        
        self.num_landers = num_landers
        self.landers = []
    
    def create_landers(self):
        for i in range(self.num_landers):
            x = random.randint(10,self.screen.get_size()[0])
            self.landers.append(Lander((x,100),self.screen,self.space,10,i))
            
    def draw_landers(self,apply_force_left,apply_force_right):
        kill_list = []
        for index,lander in enumerate(self.landers):
            if lander.has_life():
                lander.draw_and_update(apply_force_left,apply_force_right)
            else:
                kill_list.append(index)
        
        # for index in sorted(kill_list,reverse=True):
        #     self.landers.pop(index)
            
    def set_collision_true(self,ids):
        for lander in self.landers:
            if lander.id in ids:
                lander.set_collision(True)
            
    
    def lander_count(self):
        return len(self.landers)

class KeyboardSimulation:
    def __init__(self,
                 width=1280,
                 height=720,
                 gravity= 9.81,
                 terrain_exaggeration = 300,
                 terrain_corners = 500,
                 keypress_enabled=True,
                 physics_debug = False):
        
        pygame.init()
        pygame.display.set_caption('Mun Lander') 
        
        self.width   = width
        self.height  = height 
        
        self.screen  = pygame.display.set_mode((width, height))
        self.clock   = pygame.time.Clock()
        self.fps     = 60
        self.dt      = 1/self.fps
        
        self.running = True
        
        self.terrain_vertexes     = []
        self.terrain_break_count  = terrain_corners
        self.terrain_exaggeration = terrain_exaggeration
        
        self.space         = pymunk.Space()
        self.space.gravity = (0, gravity*100)
        
        self.lander_factory  = LanderFactory(self.screen,self.space,1)
        self.collion_handler = None
        
        self.keypress_enabled = keypress_enabled
        self.physics_debug    = physics_debug
        
        # Handle Keypresses for rocket
        
        self.apply_force_left  = False
        self.apply_force_right = False
        
        self.init()
    
    def init(self):    
        self.generate_terrain_vertexes()
        self.generate_terrain_physics()

        self.lander_factory.create_landers()
        
        self.collion_handler = self.space.add_collision_handler(Categories.LANDER_CAT,Categories.TERRAIN_CAT)
        self.collion_handler.post_solve = self.handle_collision
    
    def loop(self):
        self.draw_terrain()
        self.lander_factory.draw_landers(self.apply_force_left,self.apply_force_right)
    
    def end(self):
        pass
    
    def run(self):
        do = pymunk.pygame_util.DrawOptions(self.screen)
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                if event.type == pygame.KEYDOWN and self.keypress_enabled:
                    if event.key == pygame.K_UP:
                        self.apply_force_right = True
                        self.apply_force_left  = True
                    if event.key == pygame.K_LEFT:
                        self.apply_force_left = True
                    if event.key == pygame.K_RIGHT:   
                        self.apply_force_right = True
                else:
                    self.apply_force_right = False
                    self.apply_force_left  = False
                    
            self.screen.fill("black")
            
            if self.physics_debug:
                self.space.debug_draw(do)
            
            self.loop()

            pygame.display.flip()
            
            self.space.step(self.dt)        
            self.clock.tick(self.fps)

        self.end()
        pygame.quit()
        
    def generate_terrain_vertexes(self):
        terrain_break_heights = [generate_noise([x/self.terrain_break_count,0]) for x in range(self.terrain_break_count)]
        
        start_index = 0
        gap = self.width / (self.terrain_break_count - 1) 
        
        self.terrain_vertexes.append([start_index,self.height])
        
        for index, height in enumerate(terrain_break_heights[:-1]):
            height1 = min(self.height-10,(self.height * 0.8) + (height * self.terrain_exaggeration))
            p1 = [start_index, height1]

            start_index += gap

            height2 = min(self.height-10,(self.height * 0.8) + (terrain_break_heights[index + 1] * self.terrain_exaggeration))
            p2 = [start_index, height2]
            
            self.terrain_vertexes.append(p1)
            self.terrain_vertexes.append(p2)
        
        if len(terrain_break_heights) > 1:
            final_height = min(self.height-10,(self.height * 0.8) + (terrain_break_heights[-1] * self.terrain_exaggeration))
            self.terrain_vertexes.append([self.width,final_height])
            
        self.terrain_vertexes.append([self.width,self.height])
        
    def generate_terrain_physics(self):
        terrain_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.space.add(terrain_body)
        
        for a,b in pairwise(self.terrain_vertexes[1:-1]):
            v1 = a
            v2 = b
            v3 = [ b[0] , self.height ]
            v4 = [ a[0] , self.height ]
            
            shape = pymunk.Poly(terrain_body, [v1,v2,v3,v4])
            shape.filter = pymunk.ShapeFilter(categories=Categories.TERRAIN_CAT,mask=Categories.LANDER_CAT)
            shape.collision_type = Categories.TERRAIN_CAT
            shape.friction = 0.9
            self.space.add(shape)
        
    def draw_terrain(self):
        if not self.terrain_vertexes:
            return
        
        pygame.draw.polygon(self.screen, (255, 255, 255), self.terrain_vertexes)
    
    def handle_collision(self,arbiter,space,data):
        shapes = arbiter.shapes
        ids    = [] 
        
        if arbiter.is_first_contact:
            for shape in shapes:
                ids.append(shape.body.id)
            self.lander_factory.set_collision_true(ids)
            print(arbiter.total_ke)
         
class GeneticSimulation:
    def __init__(self,
                 width=1280,
                 height=720,
                 gravity= 9.81,
                 terrain_exaggeration = 300,
                 terrain_corners = 500,
                 lander_spawn_height = 100,
                 lander_mass = 10,
                 landing_window_seconds = 8,
                 generations = 1000,
                 config_path = 'neat_config.ini'):
        
        pygame.init()
        pygame.display.set_caption('Mun Lander Evolution') 
        
        self.width   = width
        self.height  = height 
        
        self.screen  = pygame.display.set_mode((width, height))
        self.clock   = pygame.time.Clock()
        self.fps     = 60
        self.dt      = 1/self.fps
        
        self.running = True
        
        self.terrain_vertexes     = []
        self.terrain_break_count  = terrain_corners
        self.terrain_exaggeration = terrain_exaggeration
        
        self.space         = pymunk.Space()
        self.space.gravity = (0, gravity*100)
        
        self.lander_spawn_height = lander_spawn_height
        self.lander_mass         = lander_mass
        self.landing_window      = landing_window_seconds
        self.landers             = None
        
        self.collion_handler = self.space.add_collision_handler(Categories.LANDER_CAT,Categories.TERRAIN_CAT)
        self.collion_handler.post_solve = self.handle_collision
        
        self.config_path      = config_path
        self.generation_count = generations
        
    def run(self):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             self.config_path)
        
        population = neat.Population(config)
        
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.Checkpointer(5))
        
        winner = population.run(self.run_simulation, self.generation_count)
        
        
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)            
        
    def run_simulation(self,genomes, config):
        self.terrain_vertexes  = []
        
        self.generate_terrain_vertexes()
        self.generate_terrain_physics()
        
        self.flattest_zone = self.find_flattest_region_center()
        
        
        self.landers  = []
        
        for genome_id, genome in genomes:
            genome.fitness = 0
            
            x = random.randint(50,self.width-50)
            y = self.lander_spawn_height
            
            self.landers.append(
                Lander((x,y),
                       self.screen,
                       self.space,
                       self.lander_mass,genome_id,
                       neat.nn.FeedForwardNetwork.create(genome,config),
                       genome,
                       self.flattest_zone)
            )
            
        print(len(self.landers))
        
        running =True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                    
            self.screen.fill('BLACK')
            self.draw_terrain()
            
            pygame.draw.circle(self.screen, (255,0,0), self.flattest_zone, 10)
            
            for lander in self.landers:
                if lander.has_life():
                    lander.draw_and_update()
            
            pygame.display.flip()
            self.space.step(self.dt)        
            self.clock.tick(self.fps)
            
            running = False
            for lander in self.landers:
                if lander.has_life():
                    if not lander.get_collision_status():
                        running = True
                    

        for lander in self.landers:
            lander.evaluate_fitness()
            
        for lander in self.landers:
            if lander.has_life():
                lander.kill()
        self.remove_terrain()
        
        fitness_list = [x.fitness for _, x in genomes]
        avg_fitness  = sum(fitness_list) / len(fitness_list)
        
        with open('fitness_results.txt', 'a') as file:
            file.write(f'{avg_fitness}\n')
            
         
    def generate_terrain_vertexes(self):
        noise_func = Noise()
        terrain_break_heights = [noise_func.generate_noise([x/self.terrain_break_count,0]) for x in range(self.terrain_break_count)]
        
        start_index = 0
        gap = self.width / (self.terrain_break_count - 1) 
        
        self.terrain_vertexes.append([start_index,self.height])
        
        for index, height in enumerate(terrain_break_heights[:-1]):
            height1 = min(self.height-10,(self.height * 0.8) + (height * self.terrain_exaggeration))
            p1 = [start_index, height1]

            start_index += gap

            height2 = min(self.height-10,(self.height * 0.8) + (terrain_break_heights[index + 1] * self.terrain_exaggeration))
            p2 = [start_index, height2]
            
            self.terrain_vertexes.append(p1)
            self.terrain_vertexes.append(p2)
        
        if len(terrain_break_heights) > 1:
            final_height = min(self.height-10,(self.height * 0.8) + (terrain_break_heights[-1] * self.terrain_exaggeration))
            self.terrain_vertexes.append([self.width,final_height])
            
        self.terrain_vertexes.append([self.width,self.height])
    
    def find_flattest_region_center(self,w=100):
        terrain_vertexes = self.terrain_vertexes
        flattest_start = 0
        min_slope_sum = float('inf')

        for i in range(len(terrain_vertexes) - 1):
            slope_sum = 0
            current_width = 0
            j = i

            while j < len(terrain_vertexes) - 1 and current_width < w:
                x1, y1 = terrain_vertexes[j]
                x2, y2 = terrain_vertexes[j + 1]

                if x2 == x1:
                    slope = 0  # Flat segment (vertical in x)
                else:
                    slope = abs((y2 - y1) / (x2 - x1))
                slope_sum += slope
                current_width += (x2 - x1)
                j += 1

            if current_width >= w and slope_sum < min_slope_sum:
                min_slope_sum = slope_sum
                flattest_start = i

        # Calculate the center point of the flattest region
        flattest_region_start = terrain_vertexes[flattest_start][0]
        current_width = 0
        j = flattest_start

        while j < len(terrain_vertexes) - 1 and current_width < w:
            current_width += (terrain_vertexes[j + 1][0] - terrain_vertexes[j][0])
            j += 1

        flattest_region_end = terrain_vertexes[j][0]
        center_x = (flattest_region_start + flattest_region_end) / 2

        # Find the y-coordinate corresponding to the center x-coordinate
        for k in range(len(terrain_vertexes) - 1):
            x1, y1 = terrain_vertexes[k]
            x2, y2 = terrain_vertexes[k + 1]

            if x1 <= center_x <= x2:
                t = (center_x - x1) / (x2 - x1)
                center_y = y1 + t * (y2 - y1)
                return (center_x, center_y)

        return None  # This should not be reached if input is valid

    
    def generate_terrain_physics(self):
        self.terrain_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.space.add(self.terrain_body)
        
        for a,b in pairwise(self.terrain_vertexes[1:-1]):
            v1 = a
            v2 = b
            v3 = [ b[0] , self.height ]
            v4 = [ a[0] , self.height ]
            
            shape = pymunk.Poly(self.terrain_body, [v1,v2,v3,v4])
            shape.filter = pymunk.ShapeFilter(categories=Categories.TERRAIN_CAT,mask=Categories.LANDER_CAT)
            shape.collision_type = Categories.TERRAIN_CAT
            shape.friction = 0.9
            self.space.add(shape)
            
    def remove_terrain(self):
        for shape in self.terrain_body.shapes:
            self.space.remove(shape)
        self.space.remove(self.terrain_body)
        
    def draw_terrain(self):
        if not self.terrain_vertexes:
            return
        
        pygame.draw.polygon(self.screen, (255, 255, 255), self.terrain_vertexes)

    def handle_collision(self,arbiter,space,data):
        shapes = arbiter.shapes
        
        if arbiter.is_first_contact:
            ids    = [] 
            for shape in shapes:
                ids.append(shape.body.id)
    
            for lander in self.landers:
                if lander.get_body_id() in ids:
                    lander.set_collision_data()
                
            
        
            