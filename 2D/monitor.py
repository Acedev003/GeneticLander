import io
import sys
import base64
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template_string

app = Flask(__name__)
csv_file_path = None

@app.route('/')
def plot_csv():
    df = pd.read_csv(csv_file_path)
    
    plt.figure()
    plt.plot(df['Run'], df['Avg Dist'], label='Avg Dist')
    plt.plot(df['Run'], df['Avg Speed'], label='Avg Speed')
    plt.plot(df['Run'], df['Avg Fitness'], label='Avg Fitness')
    plt.xlabel('Run')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Live Data Plot')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    html = '''
    <html>
    <head>
        <title>Live Data Plot</title>
        <meta http-equiv="refresh" content="60">
    </head>
    <body>
        <h4>{{csv_file_path}}</h4>
        <img src="data:image/png;base64,{{ plot_data }}" />
    </body>
    </html>
    '''
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return render_template_string(html, plot_data=plot_data,csv_file_path=csv_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flask app to plot live data from a CSV file.")
    parser.add_argument('csv_file', type=str, help="Path to the CSV file")
    parser.add_argument('port', type=int, help="Port no to use")
    args = parser.parse_args()
    
    csv_file_path = args.csv_file
    
    app.run(host='0.0.0.0', port=args.port)
