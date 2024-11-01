from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the model during initialization
model = joblib.load('linear_regression_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pred', methods=['POST'])
def pred():
    # Print to debug if the form data is received
    print("Form Data Received:", request.form)

    # Retrieve form input
    total_bedroom = request.form.get('Bedroom')
    
    if not total_bedroom:
        print("Please input a value")  # Debug print
        return render_template('index.html', result="Please input a value")

    try:
        customer_data = np.array([[float(total_bedroom)]])
        prediction = model.predict(customer_data)[0]  # Get scalar result
        print("Prediction made:", prediction)  # Debug print
    except Exception as e:
        print(f"Error in prediction: {e}")  # Debug print
        return render_template('index.html', result="Error in prediction")

    # Return the result to the template
    return render_template('index.html',result=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
