import argparse
from simulation import GeneticSimulation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GeneticLander Project")
    
    parser.add_argument('config_file', type=str, help="Path to the configuration file.")
    
    parser.add_argument('-g', '--generations', type=int, default=5000, help="Number of generations to run (default: 5000).")
    parser.add_argument('-sw', '--screen-width', type=int, default=1920, help="Width of the simulation screen (default: 1920).")
    parser.add_argument('-sh', '--screen-height', type=int, default=1080, help="Height of the simulation screen (default: 1080).")
    parser.add_argument('-x', '--headless', action='store_true', help="Run the simulation in headless mode (no GUI).")
    
    args = parser.parse_args()
    
    sim = GeneticSimulation(config_file=args.config_file,
                            generations=args.generations,
                            screen_width=args.screen_width,
                            screen_height=args.screen_height,
                            headless=args.headless)
    sim.run()