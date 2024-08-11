from perlin_noise import PerlinNoise

noise1 = PerlinNoise(octaves=1)
noise2 = PerlinNoise(octaves=6)
noise3 = PerlinNoise(octaves=12)

def generate_noise(coord):
    noise_val =  1.0 * noise1(coord)
    noise_val += 0.5 * noise2(coord)
    noise_val += 0.1 * noise3(coord)
    
    return noise_val

def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)