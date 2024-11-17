Terrain = Object.extend(Object)

function Terrain.new(self,
                     world,
                     category,
                     width,
                     height,
                     start_height,
                     terrain_delta)

    self.body = love.physics.newBody(world, 0, height - start_height, 'static')
    self.fixtures = {}
    self.category = category

    local prev_x = 0
    local prev_y = 0--height - start_height

    for x = 1, width+terrain_delta, terrain_delta do
        x = math.min(x + terrain_delta, width)
        local y =  -(love.math.noise((x / width) + love.math.random(15, 25)) * 90)

        if x > 0 then
            local edgeShape = love.physics.newEdgeShape(prev_x, prev_y, x, y)
            local fixture = love.physics.newFixture(self.body, edgeShape)
            fixture:setFriction(0.95)
            table.insert(self.fixtures, fixture)
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
        love.graphics.setColor(1, 1, 1, 1)
        love.graphics.setLineWidth(2)
        love.graphics.line(fixture:getBody():getWorldPoints(fixture:getShape():getPoints()))
    end
end