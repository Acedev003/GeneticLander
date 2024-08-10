from perlin_noise import PerlinNoise

noise1 = PerlinNoise(octaves=1)
noise2 = PerlinNoise(octaves=4)
noise3 = PerlinNoise(octaves=32)

def generate_noise(coord):
    noise_val = noise1(coord)
    noise_val += 1.0 * noise2(coord)
    noise_val += 0.1 * noise3(coord)
    
    return noise_val