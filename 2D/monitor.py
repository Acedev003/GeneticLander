import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
import argparse

def main(csv_file_path):
    # Initialize lists to hold data
    runs = []
    avg_dist = []
    avg_speed = []
    avg_fitness = []

    # Create a figure and axis for plotting
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], 'r-', label='Avg Dist')
    line2, = ax.plot([], [], 'g-', label='Avg Speed')
    line3, = ax.plot([], [], 'b-', label='Avg Fitness')
    ax.set_xlim(0, 100)  # Initial x-axis range
    ax.set_ylim(0, 1000000)  # Initial y-axis range
    ax.legend()
    ax.set_xlabel('Run')
    ax.set_ylabel('Value')
    ax.set_title('Live Data Plot')

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        return line1, line2, line3

    def update(frame):
        global runs, avg_dist, avg_speed, avg_fitness

        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        # Update the data lists
        runs = df['Run'].tolist()
        avg_dist = df['Avg Dist'].tolist()
        avg_speed = df['Avg Speed'].tolist()
        avg_fitness = df['Avg Fitness'].tolist()

        # Update the plot
        line1.set_data(runs, avg_dist)
        line2.set_data(runs, avg_speed)
        line3.set_data(runs, avg_fitness)

        # Adjust x-axis limits based on data
        ax.set_xlim(min(runs), max(runs))

        return line1, line2, line3

    # Create animation
    ani = animation.FuncAnimation(fig, update, init_func=init, interval=1000)  # Update every second

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live plot data from CSV file.")
    parser.add_argument('csv_file', type=str, help="Path to the CSV file")
    args = parser.parse_args()

    main(args.csv_file)
