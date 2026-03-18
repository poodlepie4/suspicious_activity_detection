from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load CNN model (.h5)
model = load_model("model/model.keras")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.grt("file")

        if file.filename == "":
            return "No file uploaded"

        # convert uploaded file to image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return "Invalid image file"

        # resize to training size
        img = cv2.resize(img, (32,32))

        # normalize
        img = img.astype("float32") / 255.0

        # reshape for model
        img = np.expand_dims(img, axis=0)

        # prediction
        prediction = model.predict(img, training=False)

        if prediction[0][0] > 0.5:
            result = "Suspicious Activity Detected"
        else:
            result = "Normal Activity"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return "ERROR: " + str(e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # for Render
    app.run(host="0.0.0.0", port=port)
