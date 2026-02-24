import pygame
import random

from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

@dataclass
class Segment:
    start_point: Point
    end_point: Point

class LanderSimulation:
    def __init__(self,
                 screen_size = (1280, 720),
                 terrain_height = 150,
                 land_strip_len = 100
        ):
        pygame.init()
        
        self.screen_width  = screen_size[0]
        self.screen_height = screen_size[1]
        self.screen   = pygame.display.set_mode(screen_size)
        
        self.clock    = pygame.time.Clock()

        self.land_strip_len   = land_strip_len
        self.terrain_height   = terrain_height
        terrain_gen_results   = self.generate_terrain(num_segments=300)
        self.terrain_segments = terrain_gen_results[1]
        self.landing_target   = terrain_gen_results[0]

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

    def generate_terrain(self,num_segments=3,bound=100,bound_diff=20) -> tuple[Point,list[Segment]]:
        start_y = self.screen_height - self.terrain_height

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
        
    def draw_terrain(self):
        for segment in self.terrain_segments:
                pygame.draw.line(self.screen, (255,255,255), (segment.start_point.x,segment.start_point.y), (segment.end_point.x,segment.end_point.y))
        

    def run(self):
        while True:
            self.handle_events()

            # Logic

            self.screen.fill((0,0,0))

            # Draw

            self.draw_terrain()
            
            pygame.display.flip()
            self.clock.tick(60)