WIDTH  = 1280
HEIGHT = 720 

function love.load()
    Object = require "libs/classic"
    require "terrain"
    love.window.setTitle("GeneticLander")
    love.window.setMode( WIDTH, HEIGHT, flags )

    love.physics.setMeter(6)
    world = love.physics.newWorld(0, 9.81 * 6,true)

    terrain = Terrain(world,WIDTH,HEIGHT,100,70)
end

function love.update(dt)
    terrain:update(dt)
end

function love.draw()
    terrain:draw()
end