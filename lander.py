import math
import pygame
import pymunk
import random

from pymunk import pygame_util
from primitives import Point
from pymunk import Transform

class TwinFlameLander:
    def __init__(
            self, 
            screen: pygame.Surface,
            space: pymunk.Space,
            terrain_height: int,
        ):

        self.screen = screen
        self.space  = space
        self.terrain_height = terrain_height

        self.w = 50
        self.h = 40

        self.dry_weight = 6800
        self.fuel_level = 1000
        self.leg_weight = 200

        # body_vertices = [
        #     (-self.w/2, 0),
        #     ( self.w/2, 0),
        #     ( self.w/4, self.h),
        #     (-self.w/4, self.h),
        # ]
        body_vertices = [
            (-self.w/2, -self.h/2),
            ( self.w/2, -self.h/2),
            ( self.w/4,  self.h/2),
            (-self.w/4,  self.h/2),
        ]
        bottom_y = -self.h/2

        lleg_vertices = [
            (-self.w/2, bottom_y),
            (-self.w/2, bottom_y - 10)  # angled outward looks better
        ]

        rleg_vertices = [
            ( self.w/2, bottom_y),
            ( self.w/2, bottom_y - 10)
        ]

        # lleg_vertices = [
        #     (-self.w/2, 0), (-self.w/2, -5)
        # ]

        # rleg_vertices = [
        #     (self.w/2, 0), (self.w/2, -5)
        # ]

        rect_moment = pymunk.moment_for_poly(self.dry_weight+self.fuel_level, body_vertices)
        seg1_moment = pymunk.moment_for_segment(self.leg_weight//2, lleg_vertices[0], lleg_vertices[1], 1)
        seg2_moment = pymunk.moment_for_segment(self.leg_weight//2, rleg_vertices[0], rleg_vertices[1], 1)

        total_moment = rect_moment + seg1_moment + seg2_moment

        self.body = pymunk.Body(mass=self.dry_weight + self.fuel_level,moment=total_moment ,body_type=pymunk.Body.DYNAMIC)
        self.body.position = (random.randint(100,self.screen.get_width()-100),self.terrain_height+500)

        body_shape = pymunk.Poly(self.body,body_vertices)
        lleg_shape = pymunk.Segment(self.body, lleg_vertices[0], lleg_vertices[1], 2)
        rleg_shape = pymunk.Segment(self.body, rleg_vertices[0], rleg_vertices[1], 2)

        shapes = [
            body_shape,
            lleg_shape,
            rleg_shape,
        ]

        self.space.add(self.body)
        for shape in shapes:
            shape.friction = 1
            self.space.add(shape)

        self.texture_default      = pygame.image.load("./assets/Lander.png").convert_alpha()
        self.texture_left_engine  = pygame.image.load("./assets/LanderLE.png").convert_alpha()
        self.texture_right_engine = pygame.image.load("./assets/LanderRE.png").convert_alpha()
        self.texture_both_engines = pygame.image.load("./assets/LanderLRE.png").convert_alpha()

        self.lander_texture = self.texture_default

    def point_to_pygame_point(self, point:Point):
        return Point(point.x,self.screen.get_height()-point.y)

    def update(self):
        pass

    def debug_draw(self):
        draw_options = pygame_util.DrawOptions(self.screen)
        draw_options.transform = (
            Transform(a=1, b=0, c=0, d=-1, tx=0, ty=self.screen.get_height())
        )
        self.space.debug_draw(draw_options)

    def draw(self):
        point = Point(self.body.position.x, self.body.position.y)
        point = self.point_to_pygame_point(point)

        angle_degrees = math.degrees(self.body.angle)
        rotated_image = pygame.transform.rotate(self.lander_texture, angle_degrees)
        image_rect = rotated_image.get_rect()
        image_rect.center = (point.x,point.y)

        self.screen.blit(rotated_image,image_rect)
        
        self.debug_draw()
        