from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model/model.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    
    # Get values from form
    features = [float(x) for x in request.form.values()]
    
    # Convert to numpy array
    final_features = np.array(features).reshape(1, -1)

    # Model prediction
    prediction = model.predict(final_features)

    # Convert result to readable output
    if prediction[0] == 1:
        result = "Suspicious Activity Detected"
    else:
        result = "Normal Activity"

    return render_template("index.html", prediction_text=result)

   if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
