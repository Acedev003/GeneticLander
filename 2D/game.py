import math
import pymunk
import pygame
import random

import pymunk.pygame_util
from utils import generate_noise , pairwise

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
        life_time = random.uniform(0.5, 1.0)
        self.particles.append(SmokeParticle(list(position), velocity, life_time, self.screen))

    def update_and_draw(self, dt):
        for particle in self.particles:
            particle.update(dt)
            particle.draw()

        # Remove dead particles
        self.particles = [p for p in self.particles if p.is_alive()]


class Lander:
    def __init__(self,position,screen,space,mass,id=None):
        self.id       = id
        self.screen   = screen
        self.screen_w = screen.get_size()[0]
        self.screen_h = screen.get_size()[1]
        self.space    = space
        self.alive    = True
        
        self.image = pygame.image.load("assets/Lander.png")
        self.image = self.image.convert_alpha()
        
        self.body = pymunk.Body()
        self.body.position = position
        
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
        
        space.add(self.body)
        for a,b in self.segments:
            segment = pymunk.Segment(self.body, a, b, 2)
            segment.color = (255,0,0,0)
            segment.mass = mass/(len(self.segments)+1)
            segment.filter = pymunk.ShapeFilter(categories=Categories.LANDER_CAT,mask=Categories.TERRAIN_CAT)
            segment.friction = 1
            segment.elasticity = 0
            space.add(segment)
        
        center_span_a = [44,25]#[5,25]
        center_span_b = [5,25]#[44,25]
        
        center_span = pymunk.Segment(self.body, center_span_a, center_span_b, 2)
        center_span.color = (255,0,0,0)
        center_span.mass = mass/(len(self.segments)+1)
        center_span.filter = pymunk.ShapeFilter(categories=Categories.LANDER_CAT,mask=Categories.TERRAIN_CAT)
        space.add(center_span)
        
        self.center_span = center_span
        
    def draw(self,apply_force_left,apply_force_right):
        
        angle_degrees = math.degrees(self.body.angle)
        rotated_image = pygame.transform.rotate(self.image, -angle_degrees)
        
        center_x,center_y = self.center_span.bb.center()
        
        center_span_angle          = self.center_span.body.angle
        center_span_angle_deg      = math.degrees(center_span_angle)
        center_span_angle_deg_norm = center_span_angle_deg % 360
        
        if center_span_angle_deg_norm > 270 or center_span_angle_deg_norm < 90:
            force = -10000
            fx    = force * math.sin(-self.center_span.body.angle)
            fy    = force * math.cos(-self.center_span.body.angle)
        else:
            force = 10000
            fx    = force * math.sin(-self.center_span.body.angle)
            fy    = force * math.cos(-self.center_span.body.angle)
            
        
        
        print("CSPAN_ANG_RAD:",center_span_angle)
        print("CSPAN_ANG_DEG:",center_span_angle_deg)
        print("CSPAN_ANG_DEG_NORM:",center_span_angle_deg_norm)
        print("FORCE:",force)
        print("FX",fx)
        print("FY",fy)
        print("#######")

        if apply_force_left:
            self.body.apply_force_at_local_point((fx,fy ), self.center_span.a)
            self.smoke_emitter.emit(self.body.local_to_world(self.center_span.a))

        if apply_force_right:
            self.body.apply_force_at_local_point((fx,fy ), self.center_span.b)
            self.smoke_emitter.emit(self.body.local_to_world(self.center_span.b))

        
        rotated_rect  = rotated_image.get_rect(center=(center_x,center_y))
        
        #pygame.draw.rect(self.image, (0, 255, 0), self.image.get_rect(), 1)
        
        self.screen.blit(rotated_image,rotated_rect)
        
        #pygame.draw.circle(self.screen,(0,255,0),(center_x,center_y),radius=5)
        
        self.smoke_emitter.update_and_draw(1/60)

        if self.body.position[0] > self.screen_w or self.body.position[0] < 0 or self.body.position[1] > self.screen_h or self.body.position[1] < 0:
            self.kill()
    
    
    def kill(self):
        for shape in self.body.shapes:
            self.space.remove(shape)
        self.space.remove(self.body)
        self.alive = False
        
    def has_life(self):
        return self.alive

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
                lander.draw(apply_force_left,apply_force_right)
            else:
                kill_list.append(index)
                
        for index in sorted(kill_list,reverse=True):
            self.landers.pop(index)
    
    def lander_count(self):
        return len(self.landers)

class Simulation:
    def __init__(self,
                 width=1280,
                 height=720,
                 terrain_corners = 500):
        
        pygame.init()
        
        self.width   = width
        self.height  = height 
        
        self.screen  = pygame.display.set_mode((width, height))
        self.clock   = pygame.time.Clock()
        self.fps     = 60
        self.dt      = 1/self.fps
        self.running = True
        self.apply_force_left  = False
        self.apply_force_right = False
        
        self.terrain_vertexes     = []
        self.terrain_break_count  = terrain_corners
        self.terrain_exaggeration = 300
        
        self.space         = pymunk.Space()
        self.space.gravity = (0, 981)
        
        self.lander_factory = LanderFactory(self.screen,self.space,10)
        
        self.init()
    
    def init(self):    
        self.generate_terrain_vertexes()
        self.generate_terrain_physics()

        self.lander_factory.create_landers()
    
    def loop(self):
        self.draw_terrain()
        self.lander_factory.draw_landers(self.apply_force_left,self.apply_force_right)
        print("NUM_LANDERS:",self.lander_factory.lander_count())
    
    def end(self):
        pass
    
    def run(self):
        do = pymunk.pygame_util.DrawOptions(self.screen)
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    
                if event.type == pygame.KEYDOWN:
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
            #self.space.debug_draw(do)
            
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
            shape.friction = 0.9
            self.space.add(shape)
        
    def draw_terrain(self):
        if not self.terrain_vertexes:
            return
        
        pygame.draw.polygon(self.screen, (255, 255, 255), self.terrain_vertexes)
    