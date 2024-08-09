import cv2
import numpy as np
from ursina import *
import noise
import time
from perlin_noise import PerlinNoise
from ursina.shaders import basic_lighting_shader

def calculate_normal(v1, v2, v3):
    # Calculate two edge vectors
    edge1 = np.subtract(v2, v1)
    edge2 = np.subtract(v3, v1)
    
    # Cross product of the two edge vectors
    normal = np.cross(edge1, edge2)
    
    # Normalize the normal vector
    norm = np.linalg.norm(normal)
    if norm == 0:
        return normal
    return normal / norm

noise1 = PerlinNoise(octaves=1)
noise2 = PerlinNoise(octaves=2)
noise3 = PerlinNoise(octaves=4)
noise4 = PerlinNoise(octaves=32)


def get_noise(coord):
    noise_val = noise1(coord)
    noise_val += 0.5 * noise2(coord)
    noise_val += 0.5 * noise3(coord)
    #noise_val += 0.125 * noise4(coord)
    
    return noise_val




def generate_terrain(width, height):
    vertices = []
    triangles = []
    normals = []
    
    fake_w = width//4
    fake_h = height//4
    
    
    st = time.time()
    pic = np.array([[get_noise([i/fake_w, j/fake_h]) for j in range(fake_w)] for i in range(fake_h)])
    #pic = np.array([[get_noise([i/width, j/height]) for j in range(width)] for i in range(height)])
    dat = (pic - np.min(pic)) / np.ptp(pic)
    
    dat = cv2.resize(dat, (width, height), interpolation=cv2.INTER_LINEAR)
    
    print("COMPLETE :",time.time()-st)
    
    st = time.time()
    # Create vertices
    for y in range(height):
        for x in range(width):
            c = 50
            vertices.append((x, dat[y][x] * c, y))
            normals.append(np.array([0.0, 0.0, 0.0]))  # Initialize normal vector
    print("COMPLETE :",time.time()-st)
    
    st = time.time()
    # Create triangles and accumulate normals
    for y in range(height - 1):
        for x in range(width - 1):
            i1 = x + y * width
            i2 = i1 + 1
            i3 = i1 + width
            i4 = i3 + 1
            
            triangles.append([i1, i2, i3])
            triangles.append([i2, i4, i3])
            
            # Calculate face normals
            n1 = calculate_normal(vertices[i1], vertices[i2], vertices[i3])
            n2 = calculate_normal(vertices[i2], vertices[i4], vertices[i3])
            
            # Accumulate normals for each vertex
            normals[i1] += n1
            normals[i2] += n1 + n2
            normals[i3] += n1 + n2
            normals[i4] += n2
    print("COMPLETE :",time.time()-st)
    # Normalize all vertex normals
    normals = [normal / np.linalg.norm(normal) for normal in normals]
    
    return vertices, triangles, normals

v, t, n = generate_terrain(250, 250)

app = Ursina()

mesh = Mesh(vertices=v, triangles=t, normals=n, mode='triangle',thickness=100)
ent = Entity(model=mesh,shader=basic_lighting_shader,texture="mun.png")

EditorCamera()
app.run()
