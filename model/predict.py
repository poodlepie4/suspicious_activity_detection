import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("model/model.pkl", "rb"))

def predict_activity(features):
    """
    Predict suspicious activity using the trained model
    """
    
    # Convert features into numpy array
    features = np.array(features).reshape(1, -1)
    
    # Model prediction
    prediction = model.predict(features)

    # Convert prediction to readable output
    if prediction[0] == 1:
        return "Suspicious Activity"
    else:
        return "Normal Activity"
