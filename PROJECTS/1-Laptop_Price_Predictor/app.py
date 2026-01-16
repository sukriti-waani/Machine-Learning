# Import Flask framework and required tools to handle webpages and form data
from flask import Flask, render_template, request

# Import pickle to load saved ML model and dataset
import pickle

# Import NumPy for array handling
import numpy as np

# Import pandas for dataframe handling
import pandas as pd

# Create the Flask application
app = Flask(__name__)

# Load the trained ML pipeline and dataset from pickle files
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

# Create the homepage route
# This function loads the form page and sends dropdown values to HTML
@app.route('/')
def index():
    return render_template(
        'index.html',
        companies=df['Company'].unique(),
        types=df['TypeName'].unique(),
        cpus=df['Cpu brand'].unique(),
        gpus=df['Gpu brand'].unique(),
        oss=df['os'].unique(),
        selected={}
    )

# Create the prediction route
# This function is called when the user submits the form
@app.route('/predict', methods=['POST'])
def predict():
    
    # Collect all values submitted by the user from the HTML form
    company = request.form.get('company')
    type_ = request.form.get('type')
    ram = int(request.form.get('ram') or 0)
    weight = float(request.form.get('weight') or 0)
    touchscreen = 1 if request.form.get('touchscreen') == 'Yes' else 0
    ips = 1 if request.form.get('ips') == 'Yes' else 0
    screen_size = float(request.form.get('screen_size') or 0)
    resolution = request.form.get('resolution')
    cpu = request.form.get('cpu')
    hdd = int(request.form.get('hdd') or 0)
    ssd = int(request.form.get('ssd') or 0)
    gpu = request.form.get('gpu')
    os = request.form.get('os')
    
    # Extract horizontal and vertical resolution and calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size
    
    # Combine all input values into a single array for prediction
    query = np.array([company, type_, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1,12)
    
    # Predict laptop price using trained model and convert from log scale to original price
    price = int(np.exp(pipe.predict(query)[0]))
    
    # Reload the page and display the predicted price
    return render_template(
    'index.html',
    companies=df['Company'].unique(),
    types=df['TypeName'].unique(),
    cpus=df['Cpu brand'].unique(),
    gpus=df['Gpu brand'].unique(),
    oss=df['os'].unique(),
    prediction=f"Predicted Laptop Price: ₹{price}",

    # send entered values back
    selected={
        'company': company,
        'type': type_,
        'ram': ram,
        'weight': weight,
        'touchscreen': touchscreen,
        'ips': ips,
        'screen_size': screen_size,
        'resolution': resolution,
        'cpu': cpu,
        'hdd': hdd,
        'ssd': ssd,
        'gpu': gpu,
        'os': os
    }
)


# Run the Flask application in debug mode
if __name__ == "__main__":
    app.run(debug=True)