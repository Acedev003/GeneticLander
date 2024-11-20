
class GeneticSimulation:
    def __init__(self,
                 config_file   : str,
                 generations   : int = 5000,
                 screen_width  : int = 1920,
                 screen_height : int = 1080,
                 headless      : bool = False
                 ):
        
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy" 
        
        pygame.init()
        pygame.display.set_caption('Planetary Lander Evolution')
            
        self.screen   = pygame.display.set_mode((screen_width, screen_height))
        self.width    = screen_width
        self.height   = screen_height
        self.headless = headless
            
        self.fps     = 24                               # Lower FPS boosts performance but may cause jitter
        
        self.dt      = 1/self.fps
        self.clock   = pygame.time.Clock()
        
        self.terrain_points       = []
        self.terrain_break_count  = 50                  # Number of polygons to make terrain
        self.terrain_complexity   = 200                 # Perlin noise param. Higher gives steeper variations
        self.min_terrain_altitude = 10                  # Lowest height of generated terrain
        self.terrain_screen_prcnt = 0.8                 # 0.5 to 0.8 recommended. Terrain height base as a percentage of screen.
        self.terrain_friction     = 0.9
        
        self.font_asset             = pygame.font.SysFont('Arial', 10)
        self.terrain_texture        = pygame.image.load("assets/moon.png").convert()
        self.lander_engine_off      = pygame.image.load("assets/Lander.png").convert_alpha()
        self.lander_left_engine_on  = pygame.image.load("assets/LanderLE.png").convert_alpha()
        self.lander_right_engine_on = pygame.image.load("assets/LanderRE.png").convert_alpha()
        self.lander_both_engine_on  = pygame.image.load("assets/LanderLRE.png").convert_alpha()
        
        self.gravity        = 1.625                     # Acceleration due to gravity
        
        self.space          = pymunk.Space()
        self.space.gravity  = (0, self.gravity*100)
        
        self.landers           = []
        self.lander_spawn_y    = 100                    # 100 to 500 recommended. Spawns lander at this y-coordinate
        self.no_spawn_margin_x = 500                    # Prevents any lander spawning in +- of this range
    

        self.neat_config_path = config_file       # Path to neat configuration file
    
        if "L-PR" in self.neat_config_path:
            self.lander_class = PulseRocker
            run_save_folder_suffix = "L-PR"
        elif "L-TFC" in self.neat_config_path:
            self.lander_class = TwinFlameCan
            run_save_folder_suffix = "L-TFC"
        else:
            run_save_folder_suffix = ""
    
        self.run_folder       = f'runs/{datetime.datetime.now()}-{run_save_folder_suffix}'
        self.fitness_file     = f'{self.run_folder}/fitness_data.csv'
        self.generation_count = generations
        self.run_counter      = 0
        
        print("FITNESS FILE PATH:",self.fitness_file)
        
        self.collion_handler = self.space.add_collision_handler(Categories.LANDER_CAT,Categories.TERRAIN_CAT)
        self.collion_handler.post_solve = self.handle_collision
     
    def run(self,resume_path : str = None):
        
        os.mkdir(self.run_folder)
        
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             self.neat_config_path)
        if resume_path:
            population = neat.Checkpointer().restore_checkpoint(resume_path)
        else:
            population = neat.Population(config)
        
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.Checkpointer(10,filename_prefix=f"{self.run_folder}/ckpt-"))
        
        winner = population.run(self.run_simulation, self.generation_count)
        pickle.dump(winner, open(os.path.join(self.run_folder, 'winner.pkl'), 'wb'))

        plot_stats(stats, ylog=False, view=True)
        plot_species(stats, view=True)     
    
    def run_simulation(self,genomes: list[tuple[int,neat.genome.DefaultGenome]],config):
        start_time = time.time()
        self.terrain_points = []
        self.landers        = []
        self.run_counter   += 1
        
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
                self.lander_class((x,y),
                       self.screen,
                       self.space,
                       genome_id,
                       neat.nn.FeedForwardNetwork.create(genome,config),
                       genome,
                       self.landing_zone,
                       self.terrain_points,
                       [
                           self.lander_engine_off,
                           self.lander_left_engine_on,
                           self.lander_right_engine_on,
                           self.lander_both_engine_on
                       ],
                       self.font_asset)
            )
            
        print("LANDERS_COUNT:",len(self.landers))
        
        running = True
        paused  = False
        while running:
            running = any(lander.has_life() and not lander.get_collision_status() for lander in self.landers)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                elif event.type ==pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused

            if paused:
                continue
            
            self.screen.fill('BLACK')
            if not self.headless:
                self.draw_terrain()
                pygame.draw.circle(self.screen, (0,255,0), self.landing_zone, 5)
            
            for lander in self.landers:
                lander.update()
                if not self.headless:
                    lander.draw()
            
            pygame.display.flip()
            self.space.step(self.dt)        
            self.clock.tick(self.fps)
            
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
                writer.writerow(['Run','Avg Dist','Avg Speed','Avg Fitness'])
            writer.writerow([self.run_counter,avg_distance, avg_velocity, avg_fitness])
        
        for genome_id, genome in genomes:
            for lander in self.landers:
                if lander.genome_id == genome_id:
                    genome.fitness = lander.fitness
            #print(genome.fitness)
        
        end_time = time.time()
        print('TIME FOR RUN:',end_time-start_time)
           
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
        
        pygame.gfxdraw.textured_polygon(self.screen,self.terrain_points,self.terrain_texture,0,0)
        #pygame.draw.polygon(self.screen, (255, 255, 255), self.terrain_points)
    
    def handle_collision(self,arbiter,space,data):
        shapes = arbiter.shapes
        
        if arbiter.is_first_contact:
            ids    = [] 
            for shape in shapes:
                ids.append(shape.body.id)
            
            for lander in self.landers:
                if lander.get_body_id() in ids:
                    lander.set_collided()
                
                
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
