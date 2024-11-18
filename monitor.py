import argparse
import pandas as pd
import plotille
import os
import time

def plot_csv(df, fitness_only):
    fig = plotille.Figure()
    fig.width = 40
    fig.height = 20
    fig.set_x_limits(min_=int(df['Run'].min()), max_=int(df['Run'].max()))
    
    if fitness_only:
        y_min = df['Avg Fitness'].min()
        y_max = df['Avg Fitness'].max()
    else:
        y_min = min(df['Avg Dist'].min(), df['Avg Speed'].min(), df['Avg Fitness'].min())
        y_max = max(df['Avg Dist'].max(), df['Avg Speed'].max(), df['Avg Fitness'].max())
    
    fig.set_y_limits(min_=y_min, max_=y_max)
    fig.color_mode = 'byte'

    if fitness_only:
        fig.plot(df['Run'], df['Avg Fitness'], lc=200, label='Avg Fitness')
    else:
        fig.plot(df['Run'], df['Avg Dist'], lc=100, label='Avg Dist')
        fig.plot(df['Run'], df['Avg Speed'], lc=150, label='Avg Speed')
        fig.plot(df['Run'], df['Avg Fitness'], lc=200, label='Avg Fitness')
    
    os.system('clear')  # Clear the terminal screen
    print(fig.show(legend=True))

def live_plot(csv_file_path, interval=15.0, fitness_only=False):
    while True:
        df = pd.read_csv(csv_file_path)
        plot_csv(df, fitness_only)
        time.sleep(interval)  # Wait for the specified interval before updating the plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to plot data from a run fitness file")
    parser.add_argument('csv_file', type=str, help="Path to the CSV file")
    parser.add_argument('--interval', type=float, default=15.0, help="Time interval between updates in seconds")
    parser.add_argument('--fitness-only', action='store_true', help="Plot only Avg Fitness data")
    args = parser.parse_args()
    
    live_plot(args.csv_file, args.interval, args.fitness_only)
