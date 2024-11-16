Terrain = Object.extend(Object)

function Terrain.new(self,
                     world,
                     width,
                     height,
                     start_height,
                     terrain_delta)

    self.body = love.physics.newBody(world, 0, 0, 'static')
    self.fixtures = {}

    local prev_x = 0
    local prev_y = height - start_height

    for x = 1, width+terrain_delta, terrain_delta do
        x = math.min(x + terrain_delta, width)
        local y = height - start_height - (love.math.noise((x / width) + love.math.random(5, 15)) * 90)

        if x > 0 then
            local edgeShape = love.physics.newEdgeShape(prev_x, prev_y, x, y)
            local fixture = love.physics.newFixture(self.body, edgeShape)
            table.insert(self.fixtures, fixture)
            print(prev_x, prev_y, x, y)
        end

        if x >= width then
            break
        end

    prev_x = x
    prev_y = y
    end
end

function Terrain.update(self,dt)
end

function Terrain.draw(self)
    for i=1, #self.fixtures do
        fixture = self.fixtures[i]
        love.graphics.setColor(1, 0, 0, 1)
        love.graphics.setLineWidth(2)
        print(fixture:getBody():getWorldPoints(fixture:getShape():getPoints()))
        love.graphics.line(fixture:getBody():getWorldPoints(fixture:getShape():getPoints()))
    end
end