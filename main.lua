Object = require "libs/classic"
require "terrain"
require "lander"

WIDTH  = 1280
HEIGHT = 720 

TERRAIN_PHYSICS_CATERGORY = 1
LANDERS_PHYSICS_CATERGORY = 2

function love.load()
    love.window.setTitle("GeneticLander")
    love.window.setMode( WIDTH, HEIGHT, flags )

    love.physics.setMeter(15)
    world = love.physics.newWorld(0, 1.62 * 15,true)

    terrain = Terrain(world,LANDERS_PHYSICS_CATERGORY,WIDTH,HEIGHT,70,70)

    landers = {}
    for i = 1,10 do
        local x = love.math.random( 100, 1200 )
        local l = Lander(world,TERRAIN_PHYSICS_CATERGORY,x,60)
        table.insert(landers, l)
    end
end

function love.update(dt)
    world:update(dt)
    terrain:update(dt)
    for i = 1,10 do
        landers[i]:update(dt)
    end
end

function love.draw()
    terrain:draw()
    for i = 1,10 do
        landers[i]:draw(dt)
    end
end

