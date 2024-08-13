import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the CSV file
df = pd.read_csv('fitness_results.csv')

# List of headers to plot
headers = ['Avg Roll', 'Avg Speed', 'Avg Life', 'Avg Dist', 'Avg Hor', 'Avg Fitness']

# Create a figure
fig = go.Figure()

# Add a line for each header
for header in headers:
    fig.add_trace(go.Scatter(x=df.index, y=df[header], mode='lines', name=header))

# Update layout
fig.update_layout(title='Multiple Metrics over Index',
                  xaxis_title='Index',
                  yaxis_title='Values',
                  legend_title='Metrics')

# Show the figure
fig.show()