import math
import neat
import pymunk
import pygame
import random

class TwinFlameCan:
    def __init__(self,
                 screen:pygame.Surface,
                 space:pymunk.Space,
                 terrain,
                 nn_data,
                 config):
        
        self.space   = space
        self.screen  = screen
        self.config  = config
        self.terrain = terrain
        self.nn_data = nn_data
        self.body    = pymunk.Body()
        self.body.position = (random.randint(100,self.screen.get_width()-100),int(self.config['SIMULATION']['spawn_height']))

        self.dry_weight = int(self.config['LANDER']['dry_weight'])
        self.fuel_level = int(self.config['LANDER']['fuel_level'])

        self.w, self.h = 50, 50
        vs   = [(-self.w/2+5,-self.h/2+5), (self.w/2-5,-self.h/2+5), (self.w/2,self.h/2), (-self.w/2,self.h/2)]
        
        self.shape = pymunk.Poly(self.body, vs)
        self.shape.friction = 1
        self.shape.collision_type = int(self.config['SIMULATION']['category'])
        self.shape.mass = self.dry_weight + self.fuel_level
        
        self.space.add(self.body)
        self.space.add(self.shape)

        self.texture_default      = pygame.image.load(self.config['LANDER']['texture_default']).convert_alpha()
        self.texture_left_engine  = pygame.image.load(self.config['LANDER']['texture_left_engine']).convert_alpha()
        self.texture_right_engine = pygame.image.load(self.config['LANDER']['texture_right_engine']).convert_alpha()
        self.texture_both_engine  = pygame.image.load(self.config['LANDER']['texture_both_engine']).convert_alpha()

        self.lander_texture = self.texture_default

        self.max_init_velocity = int(self.config['SIMULATION']['max_init_velocity'])
        self.max_init_angle    = int(self.config['SIMULATION']['max_init_angle_deg'])

        self.body.velocity = [random.randint(-self.max_init_velocity,self.max_init_velocity),random.randint(0,self.max_init_velocity)]
        self.body.angle    = math.radians(random.randint(-self.max_init_angle,self.max_init_angle))

        self.engine_force  = int(self.config['LANDER']['max_engine_power'])

        self.land_velocity = self.max_init_velocity

        self.alive  = True
        self.landed = False
        
        self.cause_of_death = "NA"
    
    def is_alive(self):
        return self.alive

    def kill(self,msg):
        try:
            self.space.remove(self.shape)
            self.space.remove(self.body)
        except Exception as e:
            print(e)

        self.alive = False
        self.cause_of_death = msg
    
    def find_slope_and_y(self,x,current_segment):
        x1 , y1 = current_segment[0]
        x2 , y2 = current_segment[1]
        slope = (y2 - y1) / (x2 - x1)

        y = y1 + slope * (x- x1)
        pygame.draw.circle(self.screen,'red',(x,y),4)        
        return slope,y
    
    def set_collided(self,impulse):
        if not self.landed:
            self.land_velocity = abs(self.body.velocity)
            self.landed  = True

    def update(self):
        if not self.alive: 
            return

        self.current_pos = int(self.body.position.x), int(self.body.position.y)       
        self.angle       = self.body.angle 
        self.sin_angle   = math.sin(self.angle)
        self.cos_angle   = math.cos(self.angle)
        self.angular_vel = self.body.angular_velocity
        
        self.vel_x,self.vel_y = self.body.velocity

        self.zone_dist_l = self.current_pos[0]
        self.zone_dist_r = self.screen.get_width() - self.current_pos[0] 

        if self.angular_vel > 30.0:
            self.kill("ang vel exceed")
            return
        
        self.current_segment = None
        for seg in self.terrain['segment_coords']:
            if seg[1][0] > self.current_pos[0] and self.current_pos[0] > 0:
                self.current_segment = seg
                pygame.draw.line(self.screen,'green',seg[0],seg[1],10)
                break

        if self.current_segment is None:
            self.kill("escaped the universe")
            return
        
        if self.landed:
            if self.land_velocity > int(self.config['LANDER']['max_land_vel']):
                self.kill("landed in too hot")
                return
            
        x = self.current_pos[0]
        self.slope,y = self.find_slope_and_y(x,self.current_segment)
        self.distance_to_surface =  y - self.current_pos[1] - self.h/2



        self.body.apply_force_at_local_point((0,-0),(-10,50))
        self.body.apply_force_at_local_point((0,-0),(10,50))

        
    def draw(self):
        if not self.alive: return
        pass