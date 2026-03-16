from flask import Flask, render_template, request
import numpy as np
import cv2
import pickle

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    # convert uploaded file to image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # resize to training size
    img = cv2.resize(img, (64,64))

    # normalize
    img = img / 255.0

    # reshape for model
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        result = "Suspicious Activity"
    else:
        result = "Normal Activity"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run()
