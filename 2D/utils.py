import random
from perlin_noise import PerlinNoise

noise1 = PerlinNoise(octaves=2)
noise2 = PerlinNoise(octaves=6)
noise3 = PerlinNoise(octaves=2)

def generate_noise(coord):
    noise_val =  1.0 * noise1(coord)
    noise_val += 0.7 * noise2(coord)
    noise_val += 0.1 * noise3(coord)
    
    return noise_val


class Noise:
    def __init__(self):
        self.noise1 = PerlinNoise(octaves=2,seed=random.randint(0,100))
        self.noise2 = PerlinNoise(octaves=6,seed=random.randint(0,100))
        self.noise3 = PerlinNoise(octaves=2,seed=random.randint(0,100))

    def generate_noise(self,coord):
        noise_val =  1.0 * self.noise1(coord)
        noise_val += 0.7 * self.noise2(coord)
        noise_val += 0.1 * self.noise3(coord)
    
        return noise_val

def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)