import pygame
import pymunk
import random

from primitives import Segment, Point
from lander import TwinFlameLander

class LanderSimulation:
    def __init__(self,
                 screen_size = (1280, 720),
                 terrain_height = 150,
                 land_strip_len = 100,
                 gravity = 500

        ):
        pygame.init()
        pygame.display.set_caption('Genetic Lander')
        
        self.screen_width  = screen_size[0]
        self.screen_height = screen_size[1]
        self.screen   = pygame.display.set_mode(screen_size)
        
        self.clock    = pygame.time.Clock()

        self.land_strip_len   = land_strip_len
        self.terrain_height   = terrain_height
        terrain_gen_results   = self.generate_terrain(num_segments=100)
        self.terrain_segments = terrain_gen_results[1]
        self.landing_target   = terrain_gen_results[0]

        self.gravity       = gravity
        self.physics_space = pymunk.Space()
        self.physics_space.gravity = (0, -500)

        self.init_physics_terrain()


    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

    def midpoint_displace(self,num_segments=3,start_x=0,start_y=0,end_x=0,end_y=0,bound=100, bound_diff=20) -> list[Segment]:
        segments = [
            Segment(
                start_point=Point(start_x, start_y),
                end_point=Point(end_x, end_y)
            )
        ]

        if bound_diff > bound:
            raise ValueError("bound_diff must be less than or equal to bound")

        while len(segments) < num_segments:
            new_segments  = []
            for segment in segments:
                midpoint_x = (segment.start_point.x+segment.end_point.x)//2
                midpoint_y = (segment.start_point.y+segment.end_point.y)//2 + random.randint(-bound,bound)

                midpoint = Point(midpoint_x,midpoint_y)

                new_segments.extend([
                    Segment(start_point=segment.start_point, end_point=midpoint),
                    Segment(start_point=midpoint, end_point=segment.end_point)
                ])

            segments = new_segments
            bound -= bound_diff
            bound = max(bound,1)
        
        return segments

    def point_to_pygame_point(self,point: Point):
        return Point(point.x,self.screen_height-point.y)

    def generate_terrain(self,num_segments=3,bound=100,bound_diff=20) -> tuple[Point,list[Segment]]:
        start_y = self.terrain_height

        segments = self.midpoint_displace(
            num_segments=num_segments,
            start_x=0,
            start_y=start_y,
            end_x=self.screen_width,
            end_y=start_y,
            bound=bound,
            bound_diff=bound_diff
        )

        while True:
            landing_segment_index   = random.choice(range(len(segments)))

            landing_segment_start_x = segments[landing_segment_index].start_point.x
            landing_segment_start_y = segments[landing_segment_index].start_point.y

            landing_zone_midpoint   = Point(
                (landing_segment_start_x + self.land_strip_len) // 2,
                landing_segment_start_y
            )

            landing_strip_end_x = landing_segment_start_x + self.land_strip_len
            if landing_strip_end_x > self.screen_width:
                continue

            for segment in segments[landing_segment_index:]:
                if segment.start_point.x > landing_strip_end_x:
                    segment.start_point.y = landing_segment_start_y
                    break

                segment.start_point.y = landing_segment_start_y
                segment.end_point.y   = landing_segment_start_y

            return landing_zone_midpoint, segments

    def init_physics_terrain(self):
        body  = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.physics_space.add(body)
        for segment in self.terrain_segments:
            start_point = (segment.start_point.x,segment.start_point.y)
            end_point   = (segment.end_point.x,segment.end_point.y)

            shape = pymunk.Segment(body, start_point, end_point, 5)
            shape.friction = 0.1
            self.physics_space.add(shape)

    def draw_terrain(self):
        for segment in self.terrain_segments:
                start_point = self.point_to_pygame_point(segment.start_point)
                end_point   = self.point_to_pygame_point(segment.end_point)
                pygame.draw.line(self.screen, (255,255,255), (start_point.x,start_point.y), (end_point.x,end_point.y))
        
    def run(self):
        lander = TwinFlameLander(
            self.screen,
            self.physics_space,
            self.terrain_height
        )
        while True:
            # Logic
            self.handle_events()

            self.physics_space.step(1/60)

            # Render
            self.screen.fill((0,0,0))

            self.draw_terrain()
            lander.draw()
            pygame.display.flip()
            self.clock.tick(60)