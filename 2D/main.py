
import pygame
import pymunk
import pymunk.pygame_util
from game import Lander
from utils import generate_noise


def create_ball(space,radi,mass):
    b = pymunk.Body()
    b.position = (500,40)
    shape = pymunk.Circle(b,radi)
    shape.mass = mass
    shape.color = (255,255,255,255)
    space.add(b,shape)

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)
    

class Simulation:
    def __init__(self,
                 width=1280,
                 height=720,
                 terrain_corners = 50):
        
        pygame.init()
        
        self.width   = width
        self.height  = height 
        
        self.screen  = pygame.display.set_mode((width, height))
        self.clock   = pygame.time.Clock()
        self.fps     = 60
        self.dt      = 1/self.fps
        self.running = True
        
        self.terrain_vertexes     = []
        self.terrain_break_count  = terrain_corners
        self.terrain_exaggeration = 300
        
        self.space         = pymunk.Space()
        self.space.gravity = (0, 981)
        
        self.draw_options  = pymunk.pygame_util.DrawOptions(self.screen) 
        
        self.init()
    
    def init(self):
        
        ball = create_ball(self.space,10,100)
        self.generate_terrain_vertexes()
        self.generate_terrain_physics()
    
    def loop(self):
        self.draw_terrain()
    
    def end(self):
        pass
    
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    
            self.screen.fill("black")
            self.space.debug_draw(self.draw_options)
            
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
        
        # terrain_shape = pymunk.Poly(terrain_body, self.terrain_vertexes)
        # terrain_shape.color = (25,25,255,255)
        # self.space.add(terrain_body,terrain_shape)
        d=3
        self.space.add(terrain_body)
        for a,b in pairwise(self.terrain_vertexes):
            v1 = a
            v2 = b
            v3 = [ b[0] , self.height ]
            v4 = [ a[0] , self.height ]
            
            shape = pymunk.Poly(terrain_body, [v1,v2,v3,v4])
            shape.color = (25,25,255,255)
            self.space.add(shape)
        
    def draw_terrain(self):
        if not self.terrain_vertexes:
            return
        
        #pygame.draw.polygon(self.screen, (255, 255, 255), self.terrain_vertexes)

    
if __name__ == '__main__':
    sim = Simulation()
    sim.run()