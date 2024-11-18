import math
import neat
import pymunk
import pygame
import random

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
        life_time = 1.5  # You can adjust this as needed
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

class TwinFlameCan2:
    def __init__(self,
                 screen,
                 space):
        
        self.space = space
        self.screen = screen
        self.body  = pymunk.Body()
        self.body.position = (random.randint(100,1200),50)
        w, h = 50, 50

        vs = [(-w/2,-h/2), (w/2,-h/2), (w/2,h/2), (-w/2,h/2)]

        self.shape = pymunk.Poly(self.body, vs)
        self.shape.mass = 100
        self.shape.friction = 1
        self.shape.collision_type = 1
        self.space.add(self.body)
        self.space.add(self.shape)

    def update(self):
        pass

    def draw(self):
        p = int(self.body.position.x), int(self.body.position.y)
        pygame.draw.circle(self.screen, (0,0,255), p, 10, 2)

class TwinFlameCan:
    def __init__(self,
                 position  : tuple[int,int],
                 screen    : pygame.Surface,
                 space     : pymunk.Space,
                 genome_id : int,
                 network   : neat.nn.FeedForwardNetwork,
                 genome    : neat.DefaultGenome,
                 target_zone  : tuple[int,int],
                 terrain_data : list[tuple[int,int]],
                 image_assets : list[pygame.Surface],
                 render_font  : pygame.font.Font):
        
        self.screen   = screen
        self.screen_w = screen.get_size()[0]
        self.screen_h = screen.get_size()[1]
        self.space    = space
        self.font     = render_font
        self.smoke    = SmokeEmitter(screen)
        
        ####### SPRITES #######
        
        self.lander_engine_off      = image_assets[0]
        self.lander_left_engine_on  = image_assets[1]
        self.lander_right_engine_on = image_assets[2]
        self.lander_both_engine_on  = image_assets[3]
        
        ####### FUEL, THRUST AND MASS #######
        
        self.dry_mass       = 626
        self.max_fuel       = 845
        self.fuel           = self.max_fuel
        self.thrust         = 800000
        self.consume_rate   = 2
        self.engine_force_l = 0
        self.engine_force_r = 0
        
        ####### BODY CONFIGURATION #######
        
        self.body          = pymunk.Body()
        space.add(self.body)
        
        self.segments      = [
                                ([5,10],[44,10]),
                                ([44,10],[44,38]),
                                ([44,38],[49,49]),
                                ([5,38],[1,49]),
                                ([5,38],[5,10]),
                            ]
        
        for a,b in self.segments:
            segment = pymunk.Segment(self.body, a, b, 2)
            segment.color = (255,0,0,0)
            segment.mass = self.dry_mass/(len(self.segments))
            segment.filter = pymunk.ShapeFilter(categories=Categories.LANDER_CAT,mask=Categories.TERRAIN_CAT)
            segment.collision_type = Categories.LANDER_CAT
            segment.friction = 1
            segment.elasticity = 0
            space.add(segment)

        center_fuel_span_a = [5,38]
        center_fuel_span_b = [44,38]
        
        center_fuel_span = pymunk.Segment(self.body, center_fuel_span_a, center_fuel_span_b, 2)
        center_fuel_span.color = (255,0,0,0)
        center_fuel_span.mass = 0
        center_fuel_span.filter = pymunk.ShapeFilter(categories=Categories.LANDER_CAT,mask=Categories.TERRAIN_CAT)
        center_fuel_span.collision_type = Categories.LANDER_CAT
        space.add(center_fuel_span)
        
        self.center_fuel_span = center_fuel_span
        
        ####### GENERAL PARAMS #######        

        self.body.position    = position
        self.body_id          = self.body.id
        self.genome_id        = genome_id
        self.network          = network
        self.genome           = genome 
        self.infinity_value   = 100000000  
         
        self.alive            = True
        self.has_collided     = False
         
        self.target_zone      = target_zone
        self.terrain_data     = terrain_data
        self.skin             = self.lander_engine_off
         
        self.x_pos            = self.center_fuel_span.bb.center()[0]
        self.y_pos            = self.center_fuel_span.bb.center()[1]
        self.velocity_x       = self.body.velocity[0]
        self.velocity_y       = self.body.velocity[1]
        self.killed_by_roll   = False
        self.killed_by_flying = False
        self.killed_by_speed  = False 
        self.roll_percentage  = 1.0                                      # Amount of roll [0 to 1]
        self.roll_penalty     = 0                                        # Penalty for exceeding tilt angle
        self.scan_probe_l     = 0                                        # Gives nearest terrain Left
        self.scan_probe_r     = 0                                        # Gives nearest terrain Right
        self.scanner_spacing  = 100                                      # Horizontal spacing between scan probes
        self.altitude         = 0                                        # Height of lander from terrain
        self.angular_velocity = 0                                        # Angular velocity of body
        self.x_dist_deviation = 0                                        # x deviation from target
        self.y_dist_deviation = 0                                        # y deviation from target
 
        self.max_land_vel_x   = 241.421                                  # Max tolerable landing velocity
        self.max_land_vel_y   = 241.421                                  # Max tolerable landing velocity
         
        self.fitness          = -self.infinity_value                     # Fitness Score
        self.dist_to_landing  =  self.infinity_value                     # Distance to landing zone
        self.abs_velocity     =  self.infinity_value                     # Absolute velocity
        self.angle            =  self.body.angle                         # Body tilt angle
        
    def update(self):
        self.angle = self.body.angle
        self.x_pos = self.center_fuel_span.bb.center()[0]
        self.y_pos = self.center_fuel_span.bb.center()[1]
        
        x1, y1 = self.x_pos, self.y_pos
        x2, y2 = self.target_zone
        self.dist_to_landing = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        angle_degrees            = math.degrees(self.angle)
        angle_degrees_normalized = angle_degrees % 360
        
        self.roll_percentage  = min(angle_degrees_normalized, 360 - angle_degrees_normalized) / 180
        
        if not self.has_collided:
            self.abs_velocity     = abs(self.body.velocity)
            self.velocity_x       = self.body.velocity[0]
            self.velocity_y       = self.body.velocity[1]
        
        if self.has_collided:
            self.fitness = 500 * (1 - math.exp(-0.01 * math.sqrt(self.dist_to_landing**2 + self.abs_velocity**2)))
        else:
            self.fitness = 500 * (1 - math.exp(-0.01 * math.sqrt(self.dist_to_landing**2 + self.abs_velocity**2))) + 2000 
            
        if self.killed_by_roll:
            self.fitness += self.infinity_value**2
            
        if not self.alive:
            return
        
        if self.roll_percentage > 0.5:
            self.killed_by_roll = True
            self.kill()
        
        #self.scan_probe_l, self.scan_probe_r = self.get_terrain_scanner_readings()
        
        self.x_dist_deviation = x2 - x1
        self.y_dist_deviation = y2 - y1
        self.angular_velocity = self.body.angular_velocity
        self.altitude         = self.get_altitude()
        
        if self.x_pos > self.screen_w or self.x_pos < 0 or self.y_pos > self.screen_h or self.y_pos < 0:
            self.killed_by_flying = True
            self.kill()
        
        distance_corner_r = self.screen_w - self.x_pos
        
        engine_throttle_data = self.network.activate([self.altitude,
                                                      #self.scan_probe_l,
                                                      #self.scan_probe_r,
                                                      self.x_dist_deviation,
                                                      #self.y_dist_deviation,
                                                      #distance_corner_r,
                                                      #self.x_pos,
                                                      #self.y_pos,
                                                      self.velocity_x,
                                                      self.velocity_y,
                                                      self.angular_velocity,
                                                      self.fuel
                                                      ])
        eng_l_out_raw = engine_throttle_data[0]
        eng_r_out_raw = engine_throttle_data[1]
        
        if self.fuel > 0:
            self.engine_force_l = sorted((0, (eng_l_out_raw+1)/2, 1))[1]
            self.engine_force_r = sorted((0, (eng_r_out_raw+1)/2, 1))[1]
            
            self.fuel                  -= (self.engine_force_l * self.consume_rate)
            self.fuel                  -= (self.engine_force_r * self.consume_rate)
            self.center_fuel_span.mass  = self.fuel
        else:
            self.engine_force_l = 0
            self.engine_force_r = 0
        
        
        if angle_degrees_normalized > 270 or angle_degrees_normalized < 90:
            force = -self.thrust * int(not self.has_collided)
        else:
            force = self.thrust * int(not self.has_collided)

        fx = force * math.sin(-self.angle)
        fy = force * math.cos(-self.angle)
        
        fxl = self.engine_force_l * fx
        fyl = self.engine_force_l * fy
        
        fxr = self.engine_force_r * fx
        fyr = self.engine_force_r * fy
        
        self.body.apply_force_at_local_point((fxl, fyl), self.center_fuel_span.a)
        self.body.apply_force_at_local_point((fxr, fyr), self.center_fuel_span.b)
        
    def draw(self):
        if not self.alive:
            return
        
        angle_degrees = math.degrees(self.angle)
        rotated_image = pygame.transform.rotate(self.skin, -angle_degrees)
        rotated_rect  = rotated_image.get_rect(center=(self.x_pos, self.y_pos))
        
        self.skin = self.lander_engine_off
        
        if self.engine_force_l > 0.1 and self.engine_force_r > 0.1:
            self.skin = self.lander_both_engine_on
            self.smoke.emit(self.body.local_to_world(self.center_fuel_span.a))
            self.smoke.emit(self.body.local_to_world(self.center_fuel_span.b))
        elif self.engine_force_l > 0.1:
            self.skin = self.lander_left_engine_on
            self.smoke.emit(self.body.local_to_world(self.center_fuel_span.a))
        elif self.engine_force_r > 0.1:
            self.skin = self.lander_right_engine_on
            self.smoke.emit(self.body.local_to_world(self.center_fuel_span.b))
        
        self.smoke.update_and_draw(1/30)
        
        fuel_percentage = max(0, min(1, self.fuel / self.max_fuel))
        fuel_bar_width  = 30  
        fuel_bar_height = 4  
        fuel_bar_x      = self.x_pos - fuel_bar_width / 2  
        fuel_bar_y      = self.y_pos - 40  
        
        pygame.draw.rect(self.screen, (255, 0, 0), (fuel_bar_x, fuel_bar_y, fuel_bar_width, fuel_bar_height))
        pygame.draw.rect(self.screen, (0, 255, 0), (fuel_bar_x, fuel_bar_y, fuel_bar_width * fuel_percentage, fuel_bar_height))
        
        speed_text = self.font.render(f'Vel: {self.abs_velocity:.2f}', True, (255, 255, 255))
        speed_text_rect = speed_text.get_rect(center=(fuel_bar_x, fuel_bar_y-20))

        fitness_text = self.font.render(f'Fitness: {self.fitness:.2f}', True, (255, 255, 255))
        fitness_text_rect = fitness_text.get_rect(center=(fuel_bar_x, fuel_bar_y-30))
            
        eng_text = self.font.render(f'Pow: {self.engine_force_l:.2f}L {self.engine_force_r:.2f}R', True, (255, 255, 255))
        eng_text_rect = eng_text.get_rect(center=(fuel_bar_x, fuel_bar_y-40))
        
        self.screen.blit(speed_text, speed_text_rect)
        self.screen.blit(fitness_text, fitness_text_rect)
        self.screen.blit(eng_text, eng_text_rect)
        self.screen.blit(rotated_image, rotated_rect)
       
    def set_collided(self):
        if self.has_collided:
            return

        self.has_collided = True
        if self.velocity_x > self.max_land_vel_x or self.velocity_y > self.max_land_vel_y:
            self.killed_by_speed = True
            self.kill()
        self.update()
        self.draw()
    
    def kill(self):
        try:
            for shape in self.body.shapes:
                self.space.remove(shape)

            self.space.remove(self.body)
        except:
            pass
        self.alive = False
        
    def has_life(self):
        return self.alive
    
    def get_collision_status(self):
        return self.has_collided
    
    def get_genome_id(self):
        return self.genome_id
    
    def get_body_id(self):
        return self.body_id
    
    def get_altitude(self):
        coord_x = self.x_pos
        coord_y = self.y_pos
        
        for i in range(len(self.terrain_data) - 1):
            p1 = self.terrain_data[i]
            p2 = self.terrain_data[i + 1]

            if p1[0] <= coord_x <= p2[0]:
                # Interpolate to find the terrain height at x
                y_terrain = p1[1] + (coord_x - p1[0]) * (p2[1] - p1[1]) / (p2[0] - p1[0])

                altitude = abs(coord_y - y_terrain)
                #pygame.draw.line(self.screen,(0,255,0,0.1),(self.x_pos,self.y_pos),(self.x_pos,self.y_pos+altitude))
                return  altitude
            
        return self.infinity_value
    
    def get_terrain_scanner_readings(self):
        coord_left  = (self.x_pos - self.scanner_spacing // 2, self.y_pos)
        coord_right = (self.x_pos + self.scanner_spacing // 2, self.y_pos)

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
        
        #pygame.draw.line(self.screen,(0,255,0,0.1),(self.x_pos,self.y_pos),info_left.point)
        #pygame.draw.line(self.screen,(0,255,0,0.1),(self.x_pos,self.y_pos),info_right.point)
        
        return info_left.distance if info_left else self.infinity_value, info_right.distance if info_right else self.infinity_value

    def evaluate_lander(self):
        return [self.dist_to_landing,self.abs_velocity,self.fitness]
    
class PulseRocker(TwinFlameCan):
    def __init__(self,
                 position  : tuple[int,int],
                 screen    : pygame.Surface,
                 space     : pymunk.Space,
                 genome_id : int,
                 network   : neat.nn.FeedForwardNetwork,
                 genome    : neat.DefaultGenome,
                 target_zone  : tuple[int,int],
                 terrain_data : list[tuple[int,int]],
                 image_assets : list[pygame.Surface],
                 render_font  : pygame.font.Font):
        
        super().__init__(position, screen, space, genome_id, network, genome, target_zone, terrain_data,image_assets,render_font)
        
        ####### FUEL, THRUST AND MASS #######
        
        self.dry_mass       = 626
        self.max_fuel       = 845
        self.fuel           = self.max_fuel
        self.thrust         = 900000
        self.engine_force_m = 0
        
    def update(self):
        self.angle = self.body.angle
        self.x_pos = self.center_fuel_span.bb.center()[0]
        self.y_pos = self.center_fuel_span.bb.center()[1]
        
        x1, y1 = self.x_pos, self.y_pos
        x2, y2 = self.target_zone
        self.dist_to_landing = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        angle_degrees            = math.degrees(self.angle)
        angle_degrees_normalized = angle_degrees % 360
        
        self.roll_percentage  = min(angle_degrees_normalized, 360 - angle_degrees_normalized) / 180
        
        if not self.has_collided:
            self.abs_velocity     = abs(self.body.velocity)
            self.velocity_x       = self.body.velocity[0]
            self.velocity_y       = self.body.velocity[1]
        

        if self.has_collided:
            self.fitness = 500 * (1 - math.exp(-0.01 * math.sqrt(self.dist_to_landing**2 + self.abs_velocity**2)))
        else:
            self.fitness = 500 * (1 - math.exp(-0.01 * math.sqrt(self.dist_to_landing**2 + self.abs_velocity**2))) + 2000 
            
        if self.killed_by_roll:
            self.fitness += self.infinity_value**2

            
        if not self.alive:
            return
        

        if self.roll_percentage > 0.5:
            self.killed_by_roll = True
            self.kill()
                 
        #self.scan_probe_l, self.scan_probe_r = self.get_terrain_scanner_readings()
        
        self.x_dist_deviation = x2 - x1
        self.y_dist_deviation = y2 - y1
        self.angular_velocity = self.body.angular_velocity
        self.altitude         = self.get_altitude()
        
        if self.x_pos > self.screen_w or self.x_pos < 0 or self.y_pos > self.screen_h or self.y_pos < 0:
            self.killed_by_flying = True
            self.kill()
        
        distance_corner_r = self.screen_w - self.x_pos
        
        engine_throttle_data = self.network.activate([self.altitude,
                                                      #self.scan_probe_l,
                                                      #self.scan_probe_r,
                                                      self.x_dist_deviation,
                                                      #self.y_dist_deviation,
                                                      #distance_corner_r,
                                                      #self.x_pos,
                                                      #self.y_pos,
                                                      self.velocity_x,
                                                      self.velocity_y,
                                                      self.angular_velocity,
                                                      self.fuel
                                                      ])
        eng_l_out_raw = engine_throttle_data[0]
        eng_r_out_raw = engine_throttle_data[1]
        eng_m_out_raw = engine_throttle_data[2]
        
        if self.fuel > 0:
            self.engine_force_l = sorted((0, (eng_l_out_raw+1)/2, 1))[1]
            self.engine_force_r = sorted((0, (eng_r_out_raw+1)/2, 1))[1]
            self.engine_force_m = sorted((0, (eng_m_out_raw+1)/2, 1))[1]
            
            self.fuel                  -= (self.engine_force_l * self.consume_rate/2)
            self.fuel                  -= (self.engine_force_r * self.consume_rate/2)
            self.fuel                  -= (self.engine_force_m * self.consume_rate)
            self.center_fuel_span.mass  = self.fuel
        else:
            self.engine_force_l = 0
            self.engine_force_r = 0
            self.engine_force_m = 0
        
        
        if angle_degrees_normalized > 270 or angle_degrees_normalized < 90:
            force = -self.thrust * int(not self.has_collided)
        else:
            force = self.thrust * int(not self.has_collided)

        fxm = force * math.sin(-self.angle)
        fym = force * math.cos(-self.angle)
        
        fxr = fym     / 2
        fyr = -fxm    / 2
        
        fxl = -fym    / 2
        fyl = fxm     / 2
        
        fxm = self.engine_force_m * fxm
        fym = self.engine_force_m * fym
        fxl = self.engine_force_l * fxl
        fyl = self.engine_force_l * fyl
        fxr = self.engine_force_r * fxr
        fyr = self.engine_force_r * fyr

        self.body.apply_force_at_local_point((fxm, fym), self.center_fuel_span.center_of_gravity)
        self.body.apply_force_at_local_point((fxl, fyl), self.body.center_of_gravity)
        self.body.apply_force_at_local_point((fxr, fyr), self.body.center_of_gravity)
        
    def draw(self):
        if not self.alive:
            return
        
        angle_degrees = math.degrees(self.angle)
        rotated_image = pygame.transform.rotate(self.skin, -angle_degrees)
        rotated_rect  = rotated_image.get_rect(center=(self.x_pos, self.y_pos))
        
        self.skin = self.lander_engine_off
        
        if self.engine_force_l > 0.1 and self.engine_force_r > 0.1:
            self.skin = self.lander_both_engine_on
            self.smoke.emit(self.body.local_to_world(self.center_fuel_span.a))
            self.smoke.emit(self.body.local_to_world(self.center_fuel_span.b))
        elif self.engine_force_l > 0.1:
            self.skin = self.lander_left_engine_on
            self.smoke.emit(self.body.local_to_world(self.center_fuel_span.a))
        elif self.engine_force_r > 0.1:
            self.skin = self.lander_right_engine_on
            self.smoke.emit(self.body.local_to_world(self.center_fuel_span.b))
        
        if self.engine_force_m > 0.1:
            self.skin = self.lander_both_engine_on
            self.smoke.emit(self.body.local_to_world(self.center_fuel_span.center_of_gravity))
        
        self.smoke.update_and_draw(1/30)
        
        fuel_percentage = max(0, min(1, self.fuel / self.max_fuel))
        fuel_bar_width  = 30  
        fuel_bar_height = 4  
        fuel_bar_x      = self.x_pos - fuel_bar_width / 2  
        fuel_bar_y      = self.y_pos - 40  
        
        pygame.draw.rect(self.screen, (255, 0, 0), (fuel_bar_x, fuel_bar_y, fuel_bar_width, fuel_bar_height))
        pygame.draw.rect(self.screen, (0, 255, 0), (fuel_bar_x, fuel_bar_y, fuel_bar_width * fuel_percentage, fuel_bar_height))
        
        speed_text = self.font.render(f'Vel: {self.abs_velocity:.2f}', True, (255, 255, 255))
        speed_text_rect = speed_text.get_rect(center=(fuel_bar_x, fuel_bar_y-20))

        fitness_text = self.font.render(f'Fitness: {self.fitness:.2f}', True, (255, 255, 255))
        fitness_text_rect = fitness_text.get_rect(center=(fuel_bar_x, fuel_bar_y-30))
            
        eng_text = self.font.render(f'Pow: {self.engine_force_l:.2f}L {self.engine_force_r:.2f}R', True, (255, 255, 255))
        eng_text_rect = eng_text.get_rect(center=(fuel_bar_x, fuel_bar_y-40))
        
        self.screen.blit(speed_text, speed_text_rect)
        self.screen.blit(fitness_text, fitness_text_rect)
        self.screen.blit(eng_text, eng_text_rect)
        self.screen.blit(rotated_image, rotated_rect)