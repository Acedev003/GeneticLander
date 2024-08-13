import csv
import math
import time
import neat
import pymunk
import pygame
import random
import logging
import visualize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from utils import generate_noise , pairwise , Noise

class Categories:
    LANDER_CAT  = 0b01
    TERRAIN_CAT = 0b10

class SmokeParticle:
    def __init__(self, position, velocity, life_time, surface):
        self.position = position
        self.velocity = velocity
        self.life_time = life_time
        self.initial_life_time = life_time
        self.surface = surface

    def update(self, dt):
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt
        self.life_time -= dt

    def draw(self, screen):
        if self.life_time > 0:
            alpha = max(0, int(255 * (self.life_time / self.initial_life_time)))
            self.surface.set_alpha(alpha)
            screen.blit(self.surface, (int(self.position[0]), int(self.position[1])))

    def is_alive(self):
        return self.life_time > 0
    
class SmokeEmitter:
    def __init__(self, screen):
        self.particles = []
        self.screen = screen
        self.particle_surface = pygame.Surface((5, 5), pygame.SRCALPHA)
        pygame.draw.circle(self.particle_surface, (255, 255, 255), (2, 2), 2)

    def emit(self, position):
        velocity = [random.uniform(-1, 1), random.uniform(-2, 0)]
        life_time = 0.1  # You can adjust this as needed
        self.particles.append(SmokeParticle(list(position), velocity, life_time, self.particle_surface))

    def update_and_draw(self, dt):
        # Update and draw particles
        i = 0
        while i < len(self.particles):
            particle = self.particles[i]
            particle.update(dt)
            if particle.is_alive():
                particle.draw(self.screen)
                i += 1
            else:
                self.particles.pop(i) 

class Lander:
    def __init__(self,position,screen,space,id=None,network=None,genome=None,target_zone=None,logger=None):
        
        self.screen   = screen
        self.screen_w = screen.get_size()[0]
        self.screen_h = screen.get_size()[1]
        self.space    = space
        
        self.lander_no_engine = pygame.image.load("assets/Lander.png")
        self.lander_no_engine = self.lander_no_engine.convert_alpha()
        
        self.lander_left_engine = pygame.image.load("assets/LanderLE.png")
        self.lander_left_engine = self.lander_left_engine.convert_alpha()
        
        self.lander_right_engine = pygame.image.load("assets/LanderRE.png")
        self.lander_right_engine = self.lander_right_engine.convert_alpha()
        
        self.lander_both_engine = pygame.image.load("assets/LanderLRE.png")
        self.lander_both_engine = self.lander_both_engine.convert_alpha()
        
        self.logger = logger
        
        self.body = pymunk.Body()
        self.body.position = position
        self.genome_id     = id
        self.body_id       = self.body.id
        self.network       = network
        self.genome        = genome 
        self.target_zone   = target_zone
        
        self.image = self.lander_no_engine
        
        self.altimer_scan_length = 100
        
        space.add(self.body)
        
        self.logger.debug(f'{self.genome_id}-{self.body_id} Create Lander')
        self.logger.debug(f'{self.genome_id}-{self.body_id} Target Zone: {self.target_zone}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} Altimeter Length: {self.altimer_scan_length}')
        
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
        
        self.dry_mass = 10
        
        for a,b in self.segments:
            segment = pymunk.Segment(self.body, a, b, 2)
            segment.color = (255,0,0,0)
            segment.mass = self.dry_mass/(len(self.segments))
            segment.filter = pymunk.ShapeFilter(categories=Categories.LANDER_CAT,mask=Categories.TERRAIN_CAT)
            segment.collision_type = Categories.LANDER_CAT
            segment.friction = 1
            segment.elasticity = 0
            space.add(segment)
        
        self.logger.debug(f'{self.genome_id}-{self.body_id} Add Segments')
        
        center_span_a = [5,25]
        center_span_b = [44,25]
        
        center_span = pymunk.Segment(self.body, center_span_a, center_span_b, 2)
        center_span.color = (255,0,0,0)
        center_span.mass = 0
        center_span.filter = pymunk.ShapeFilter(categories=Categories.LANDER_CAT,mask=Categories.TERRAIN_CAT)
        center_span.collision_type = Categories.LANDER_CAT
        space.add(center_span)
        
        self.center_span = center_span
        
        self.logger.debug(f'{self.genome_id}-{self.body_id} Add Center Span')
        
        ## Input Parameters
        
        self.logger.debug(f'{self.genome_id}-{self.body_id} Setting Initial Parameters')
        
        self.roll_percentage  = 1.0
        self.velocity         = 0
        self.x_pos            = self.center_span.bb.center()[0]
        self.y_pos            = self.center_span.bb.center()[1]
        self.left_alt_probe   = 0
        self.right_alt_probe  = 0
        self.max_fuel         = 400
        self.fuel             = self.max_fuel
        self.center_span.mass = self.max_fuel
    
        self.alive              = True
        self.has_collided       = False
        self.collision_velocity = 10000000
        self.collision_x = 10000000
        self.collision_y = 10000000
        self.max_collision_velocity = 500
        
        self.roll_penalty = 0
        
        x2, y2 = self.target_zone
        self.distance_to_trgt_from_collision = math.sqrt((x2 - self.collision_x)**2 + (y2 - self.collision_y)**2)
        
        self.logger.debug(f'{self.genome_id}-{self.body_id} ROLL_PERCENT         : {self.roll_percentage}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} VELOCITY             : {self.velocity}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} X_POS                : {self.x_pos}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} Y_POS                : {self.y_pos}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} LEFT_ALT_PROBE       : {self.left_alt_probe}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} RIGHT_ALT_PROBE      : {self.right_alt_probe}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} ALIVE                : {self.alive}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} HAS_COLLIDED         : {self.has_collided}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} COLLISION_VELOCITY   : {self.collision_velocity}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} COLLISION_X          : {self.collision_x}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} COLLISION_Y          : {self.collision_y}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} MAX_COLLISION_VELOCITY : {self.max_collision_velocity}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} ROLL_PENALTY         : {self.roll_penalty}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} DISTANCE_TO_TRGT_FROM_COLLISION : {self.distance_to_trgt_from_collision}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} DRY_MASS             : {self.dry_mass}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} WET_MASS             : {self.dry_mass + self.center_span.mass}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} FUEL                 : {self.fuel}')
        
    def draw_and_update(self):
        self.logger.debug(f'{self.genome_id}-{self.body_id} DRAW_AND_UPDATE() START')

        # Calculate angle in degrees and rotate image
        angle_degrees = math.degrees(self.body.angle)
        self.logger.debug(f'{self.genome_id}-{self.body_id} ANGLE_DEGREES          : {angle_degrees}')

        rotated_image = pygame.transform.rotate(self.image, -angle_degrees)
        self.logger.debug(f'{self.genome_id}-{self.body_id} ROTATED_IMAGE_APPLIED  : True')

        # Get the center of the bounding box
        center_x, center_y = self.center_span.bb.center()
        self.logger.debug(f'{self.genome_id}-{self.body_id} CENTER_X              : {center_x}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} CENTER_Y              : {center_y}')

        # Calculate angles for the center span
        center_span_angle = self.center_span.body.angle
        center_span_angle_deg = math.degrees(center_span_angle)
        self.logger.debug(f'{self.genome_id}-{self.body_id} CENTER_SPAN_ANGLE_DEG  : {center_span_angle_deg}')

        center_span_angle_deg_norm = center_span_angle_deg % 360
        self.logger.debug(f'{self.genome_id}-{self.body_id} CENTER_SPAN_ANGLE_DEG_NORM : {center_span_angle_deg_norm}')

        # Determine the force based on the normalized angle
        if center_span_angle_deg_norm > 270 or center_span_angle_deg_norm < 90:
            force = -300000 * int(not self.has_collided)
            self.logger.debug(f'{self.genome_id}-{self.body_id} FORCE_APPLIED        : {force}')
        else:
            force = 300000 * int(not self.has_collided)
            self.logger.debug(f'{self.genome_id}-{self.body_id} FORCE_APPLIED        : {force}')

        # Calculate force components fx and fy
        fx = force * math.sin(-self.center_span.body.angle)
        fy = force * math.cos(-self.center_span.body.angle)
        self.logger.debug(f'{self.genome_id}-{self.body_id} FORCE_X               : {fx}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} FORCE_Y               : {fy}')

        # Get altimeter readings
        alt_l, alt_r = self.get_altimeter_readings()
        self.logger.debug(f'{self.genome_id}-{self.body_id} ALT_LEFT              : {alt_l}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} ALT_RIGHT             : {alt_r}')

        # Calculate roll percentage
        self.roll_percentage = min(center_span_angle_deg_norm, 360 - center_span_angle_deg_norm) / 180
        self.logger.debug(f'{self.genome_id}-{self.body_id} ROLL_PERCENTAGE       : {self.roll_percentage}')

        # Calculate velocity and position values
        self.velocity = abs(self.body.velocity)
        self.logger.debug(f'{self.genome_id}-{self.body_id} VELOCITY              : {self.velocity}')

        self.velocity_x = self.body.velocity[0]
        self.velocity_y = self.body.velocity[1]
        self.logger.debug(f'{self.genome_id}-{self.body_id} VELOCITY_X            : {self.velocity_x}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} VELOCITY_Y            : {self.velocity_y}')

        self.x_pos = self.center_span.bb.center()[0]
        self.y_pos = self.center_span.bb.center()[1]
        self.logger.debug(f'{self.genome_id}-{self.body_id} X_POS                 : {self.x_pos}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} Y_POS                 : {self.y_pos}')

        self.x_pos_percentage = self.x_pos / self.screen_w
        self.y_pos_percentage = self.y_pos / self.screen_h
        self.logger.debug(f'{self.genome_id}-{self.body_id} X_POS_PERCENTAGE      : {self.x_pos_percentage}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} Y_POS_PERCENTAGE      : {self.y_pos_percentage}')

        self.left_alt_probe = alt_l
        self.right_alt_probe = alt_r
        self.logger.debug(f'{self.genome_id}-{self.body_id} LEFT_ALT_PROBE        : {self.left_alt_probe}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} RIGHT_ALT_PROBE       : {self.right_alt_probe}')

        # Calculate angular velocity
        self.angular_velocity = self.body.angular_velocity
        self.logger.debug(f'{self.genome_id}-{self.body_id} ANGULAR_VELOCITY      : {self.angular_velocity}')

        # Calculate distance to target
        x1, y1 = self.x_pos, self.y_pos
        x2, y2 = self.target_zone
        self.distance_to_trgt = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        self.logger.debug(f'{self.genome_id}-{self.body_id} DISTANCE_TO_TRGT      : {self.distance_to_trgt}')

        # Calculate directional components towards the target
        self.to_left_right = x2 - x1
        self.to_up_down = y2 - y1
        self.logger.debug(f'{self.genome_id}-{self.body_id} TO_LEFT_RIGHT         : {self.to_left_right}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} TO_UP_DOWN            : {self.to_up_down}')

        # Apply roll penalty if applicable
        if self.roll_percentage > 0.5:
            self.roll_penalty += 100000
            self.logger.debug(f'{self.genome_id}-{self.body_id} ROLL_PENALTY_APPLIED : {self.roll_penalty}')

        engine_data    = self.network.activate([self.left_alt_probe,
                                                self.right_alt_probe,
                                                self.to_left_right,
                                                self.to_up_down,
                                                self.x_pos,
                                                self.y_pos,
                                                self.velocity_x,
                                                self.velocity_y,
                                                self.angular_velocity,
                                                math.sin(center_span_angle),
                                                math.cos(center_span_angle),
                                                #center_span_angle_deg_norm,
                                                self.fuel],)
        
        # Extract engine forces
        if self.fuel >= 0:
            engine_force_l = max(0,engine_data[0])
            engine_force_r = max(0,engine_data[1])
            self.fuel     -= engine_force_l
            self.fuel     -= engine_force_r
            self.center_span.mass -= engine_force_l
            self.center_span.mass -= engine_force_r
            
            self.logger.debug(f'{self.genome_id}-{self.body_id} FUEL                 : {self.fuel}')
            self.logger.debug(f'{self.genome_id}-{self.body_id} WET_MASS             : {self.dry_mass + self.center_span.mass}')
        else:
            engine_force_l = 0
            engine_force_r = 0
            self.logger.debug(f'{self.genome_id}-{self.body_id} FUEL EMPTY - ENGINE DEAD')
        
        self.logger.debug(f'{self.genome_id}-{self.body_id} ENGINE_FORCE_L        : {engine_force_l}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} ENGINE_FORCE_R        : {engine_force_r}')

        # Calculate forces applied by engines
        fxl = engine_force_l * fx
        fyl = engine_force_l * fy
        self.logger.debug(f'{self.genome_id}-{self.body_id} FORCE_X_LEFT          : {fxl}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} FORCE_Y_LEFT          : {fyl}')

        fxr = engine_force_r * fx
        fyr = engine_force_r * fy
        self.logger.debug(f'{self.genome_id}-{self.body_id} FORCE_X_RIGHT         : {fxr}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} FORCE_Y_RIGHT         : {fyr}')

        # Apply forces at local points on the body
        self.body.apply_force_at_local_point((fxl, fyl), self.center_span.a)
        self.logger.debug(f'{self.genome_id}-{self.body_id} FORCE_APPLIED_LEFT    : {(fxl, fyl)} at {self.center_span.a}')

        self.body.apply_force_at_local_point((fxr, fyr), self.center_span.b)
        self.logger.debug(f'{self.genome_id}-{self.body_id} FORCE_APPLIED_RIGHT   : {(fxr, fyr)} at {self.center_span.b}')

        # Set image based on engine force
        self.image = self.lander_no_engine
        self.logger.debug(f'{self.genome_id}-{self.body_id} IMAGE_SET             : Lander No Engine')

        if engine_force_l > 0.1:
            self.image = self.lander_left_engine
            self.logger.debug(f'{self.genome_id}-{self.body_id} IMAGE_SET             : Lander Left Engine')
            #self.smoke_emitter.emit(self.body.local_to_world(self.center_span.a))

        if engine_force_r > 0.1:
            self.image = self.lander_right_engine
            self.logger.debug(f'{self.genome_id}-{self.body_id} IMAGE_SET             : Lander Right Engine')
            #self.smoke_emitter.emit(self.body.local_to_world(self.center_span.b))

        if engine_force_l > 0.1 and engine_force_r > 0.1:
            self.image = self.lander_both_engine
            self.logger.debug(f'{self.genome_id}-{self.body_id} IMAGE_SET             : Lander Both Engines')

        # Calculate rotated image's rectangle
        rotated_rect = rotated_image.get_rect(center=(center_x, center_y))
        self.logger.debug(f'{self.genome_id}-{self.body_id} ROTATED_RECT          : {rotated_rect}')

        # Draw the image if alive
        if self.alive:
            fuel_percentage = max(0, min(1, self.fuel / self.max_fuel))
            
            bar_width = 30  # Width of the health bar
            bar_height = 4  # Height of the health bar
            bar_x = center_x - bar_width / 2  # Center the bar above the lander
            bar_y = center_y - 40  # Position the bar above the lander

            # Draw the health bar background (red)
            pygame.draw.rect(self.screen, (255, 0, 0), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y, bar_width * fuel_percentage, bar_height))
            
            self.screen.blit(rotated_image, rotated_rect)
            self.logger.debug(f'{self.genome_id}-{self.body_id} IMAGE_BLIT           : True')
            #self.smoke_emitter.update_and_draw(1/60)

        # Check if the body is out of screen bounds
        if self.body.position[0] > self.screen_w or self.body.position[0] < 0 or self.body.position[1] > self.screen_h or self.body.position[1] < 0:
            self.kill()
            self.logger.debug(f'{self.genome_id}-{self.body_id} OUT_OF_BOUNDS        : True, Killed')

        self.logger.debug(f'{self.genome_id}-{self.body_id} DRAW_AND_UPDATE() END')
       
    def set_collision_data(self):
        self.logger.debug(f'{self.genome_id}-{self.body_id} SET_COLLISION_DATA() START')
        if self.has_collided:
            self.logger.debug(f'{self.genome_id}-{self.body_id} SET_COLLISION_DATA() END')
            return

        # Mark as collided
        self.has_collided = True
        self.logger.debug(f'{self.genome_id}-{self.body_id} HAS_COLLIDED         : {self.has_collided}')

        # Set collision data
        self.collision_velocity = self.velocity
        self.collision_x = self.x_pos
        self.collision_y = self.y_pos
        self.logger.debug(f'{self.genome_id}-{self.body_id} COLLISION_VELOCITY   : {self.collision_velocity}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} COLLISION_X          : {self.collision_x}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} COLLISION_Y          : {self.collision_y}')

        # Calculate distance to target from collision point
        x2, y2 = self.target_zone
        self.distance_to_trgt_from_collision = math.sqrt((x2 - self.collision_x)**2 + (y2 - self.collision_y)**2)
        self.logger.debug(f'{self.genome_id}-{self.body_id} DISTANCE_TO_TRGT_FROM_COLLISION : {self.distance_to_trgt_from_collision}')

        # Check if collision velocity exceeds maximum allowed, and kill if so
        if self.collision_velocity > self.max_collision_velocity:
            self.kill()
            self.logger.debug(f'{self.genome_id}-{self.body_id} COLLISION_EXCEEDS_MAX_VELOCITY : Killed due to high impact')
        
        self.logger.debug(f'{self.genome_id}-{self.body_id} SET_COLLISION_DATA() END')
    
    def kill(self):
        self.logger.debug(f'{self.genome_id}-{self.body_id} KILL() START')
        # Remove shapes and body from the space
        for shape in self.body.shapes:
            self.space.remove(shape)
            self.logger.debug(f'{self.genome_id}-{self.body_id} SHAPE_REMOVED        : {shape}')

        self.space.remove(self.body)
        self.logger.debug(f'{self.genome_id}-{self.body_id} BODY_REMOVED         : {self.body}')

        # Mark as not alive
        self.alive = False
        self.logger.debug(f'{self.genome_id}-{self.body_id} ALIVE                : {self.alive}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} KILL() END')
        
    def has_life(self):
        return self.alive
    
    def get_collision_status(self):
        return self.has_collided
    
    def get_genome_id(self):
        return self.genome_id
    
    def get_body_id(self):
        return self.body_id
    
    def get_altimeter_readings(self):
        self.logger.debug(f'{self.genome_id}-{self.body_id} GET_ALTIMETER_READINGS() START')

        # Calculate coordinates for left and right altimeter scans
        coord_left  = (self.x_pos - self.altimer_scan_length // 2, self.y_pos)
        coord_right = (self.x_pos + self.altimer_scan_length // 2, self.y_pos)
        self.logger.debug(f'{self.genome_id}-{self.body_id} COORD_LEFT           : {coord_left}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} COORD_RIGHT          : {coord_right}')

        # Query nearest points for altimeter readings
        info_left = self.space.point_query_nearest(
            coord_left, 
            self.screen_h,
            shape_filter=pymunk.ShapeFilter(categories=Categories.LANDER_CAT, mask=Categories.TERRAIN_CAT)
        )
        info_right = self.space.point_query_nearest(
            coord_right,
            self.screen_h,
            shape_filter=pymunk.ShapeFilter(categories=Categories.LANDER_CAT, mask=Categories.TERRAIN_CAT)
        )
        self.logger.debug(f'{self.genome_id}-{self.body_id} INFO_LEFT_DISTANCE   : {info_left.distance if info_left else None}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} INFO_RIGHT_DISTANCE  : {info_right.distance if info_right else None}')

        self.logger.debug(f'{self.genome_id}-{self.body_id} GET_ALTIMETER_READINGS() END')

        return info_left.distance if info_left else float('inf'), info_right.distance if info_right else float('inf')

    def evaluate_fitness(self):
        self.logger.debug(f'{self.genome_id}-{self.body_id} EVALUATE_FITNESS() START')

        # Calculate roll score
        roll_score = (1 - self.roll_percentage) ** 2
        self.logger.debug(f'{self.genome_id}-{self.body_id} ROLL_SCORE           : {roll_score}')

        # Calculate speed score based on collision velocity
        if self.collision_velocity > self.max_collision_velocity:
            speed_score = 0
        else:
            speed_score = self.max_collision_velocity - self.collision_velocity
        self.logger.debug(f'{self.genome_id}-{self.body_id} SPEED_SCORE_BASE     : {speed_score}')

        # Cubing speed score and squaring velocity for final scoring
        speed_score = speed_score ** 2
        speed_score2 = self.velocity ** 2
        self.logger.debug(f'{self.genome_id}-{self.body_id} SPEED_SCORE_CUBED    : {speed_score}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} VELOCITY_SQUARED     : {speed_score2}')

        # Life bonus and distance/horizontal scores
        life_bonus = float(self.alive)
        dist_score = self.distance_to_trgt_from_collision ** 2
        hor_score = self.to_left_right ** 2
        self.logger.debug(f'{self.genome_id}-{self.body_id} LIFE_BONUS           : {life_bonus}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} DIST_SCORE           : {dist_score}')
        self.logger.debug(f'{self.genome_id}-{self.body_id} HOR_SCORE            : {hor_score}')

        # Calculate total fitness
        total_fitness = (roll_score + speed_score - dist_score - hor_score) * life_bonus - speed_score2 - self.roll_penalty
        self.logger.debug(f'{self.genome_id}-{self.body_id} TOTAL_FITNESS        : {total_fitness}')

        # Assign fitness to genome
        self.genome.fitness = total_fitness

        self.logger.debug(f'{self.genome_id}-{self.body_id} EVALUATE_FITNESS() END')

        return roll_score, speed_score, life_bonus, dist_score, hor_score, total_fitness

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
                 landing_window_seconds = 8,
                 generations = 5000,
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
        self.landing_window      = landing_window_seconds
        self.landers             = None
        
        self.collion_handler = self.space.add_collision_handler(Categories.LANDER_CAT,Categories.TERRAIN_CAT)
        self.collion_handler.post_solve = self.handle_collision
        
        self.config_path      = config_path
        self.generation_count = generations
        
        now = time.time()
        self.logger = logging.getLogger(f'{__name__}:{now}')
        #logging.basicConfig(filename=f'runs/run-{now}.log', encoding='utf-8', level=logging.DEBUG)
        
    def run(self):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             self.config_path)
        
        population = neat.Population(config)
        
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.Checkpointer(50,filename_prefix="checkpoints/ckpt-"))
        
        winner = population.run(self.run_simulation, self.generation_count)
        
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)            
        
    def run_simulation(self,genomes, config):
        self.logger.debug('NEW_SIMULATION START')
        self.terrain_vertexes  = []
        
        self.generate_terrain_vertexes()
        self.generate_terrain_physics()
        
        self.flattest_zone = self.find_flattest_region_center()
        
        self.landers  = []
        
        for genome_id, genome in genomes:
            genome.fitness = 0
            
            x = random.randint(50,self.width-50)
            while True:
                target_x = self.flattest_zone[0]
                if target_x - 200 < x < target_x + 200:
                    x = random.randint(50,self.width-50)
                else:
                    break
     
            y = self.lander_spawn_height
            
            self.landers.append(
                Lander((x,y),
                       self.screen,
                       self.space,
                       genome_id,
                       neat.nn.FeedForwardNetwork.create(genome,config),
                       genome,
                       self.flattest_zone,
                       self.logger)
            )
            
        print("LANDERS_COUNT:",len(self.landers))
        
        # Pre Loop
        running =True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                    
            self.screen.fill('BLACK')
            self.draw_terrain()
            
            pygame.draw.circle(self.screen, (255,0,0), self.flattest_zone, 5)
            
            for lander in self.landers:
                if lander.has_life():
                    lander.draw_and_update()
            
            pygame.display.flip()
            self.space.step(self.dt)        
            self.clock.tick(self.fps)
            
            running = False
            for lander in self.landers:
                if lander.has_life():
                    if  lander.get_collision_status() == False:
                        running = True
            
        # Post Loop   
        roll_sum, speed_sum, life_sum, dist_sum, hor_sum, fitness_sum = 0, 0, 0, 0, 0, 0
        for lander in self.landers:
            r,s,l,d,h,t = lander.evaluate_fitness()
            if lander.has_life():
                lander.kill()   
            
            # Sum the values for averaging
            roll_sum += r
            speed_sum += s
            life_sum += l
            dist_sum += d
            hor_sum += h
            fitness_sum += t
            
        self.remove_terrain()
        
        num_landers = len(self.landers)
        avg_roll = roll_sum / num_landers if num_landers > 0 else 0
        avg_speed = speed_sum / num_landers if num_landers > 0 else 0
        avg_life = life_sum / num_landers if num_landers > 0 else 0
        avg_dist = dist_sum / num_landers if num_landers > 0 else 0
        avg_hor = hor_sum / num_landers if num_landers > 0 else 0
        avg_fitness = fitness_sum / num_landers if num_landers > 0 else 0
        
        with open('fitness_results.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['Avg Roll', 'Avg Speed', 'Avg Life', 'Avg Dist', 'Avg Hor', 'Avg Fitness'])
            writer.writerow([avg_roll, avg_speed, avg_life, avg_dist, avg_hor, avg_fitness])
               
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
                
            
        
            