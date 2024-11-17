Lander = Object.extend(Object)

function Lander.new(self,world,category,spawn_x,spawn_y)
    self.body    = love.physics.newBody(world, spawn_x, spawn_y, 'dynamic')
    self.shape   = love.physics.newRectangleShape(50, 50)
    self.fixture = love.physics.newFixture(self.body, self.shape)

    self.image   = love.graphics.newImage("assets/Lander.png")
    
    self.fixture:setGroupIndex( -10 )
    self.body:setMass(10000)

    self.alive = true
end

function Lander.update(self)
    
end

function Lander.draw(self)

    love.graphics.setColor(1, 1, 1, 1)
    love.graphics.draw(
        self.image,
        self.body:getX(),
        self.body:getY(),
        self.body:getAngle(),
        1,1,50/2,50/2
    ) 
end