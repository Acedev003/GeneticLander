import argparse
from simulation import GeneticSimulation2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Genetic Lander")
    
    parser.add_argument('-cs', '--config_simulation', type=str, default="configs/simulation.ini", help="Path to simulation config")
    parser.add_argument('-cl', '--config_lander', type=str, default="configs/lander.ini", help="Path to lander config")
    parser.add_argument('-ct', '--config_terrain', type=str, default="configs/terrain.ini", help="Path to terrain config")
    
    args = parser.parse_args()
    sim = GeneticSimulation2(
        simulation_config_file=args.config_simulation,
        lander_config_file=args.config_lander,
        terrain_config_file=args.config_terrain,
    )

    sim.run()