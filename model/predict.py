from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load model
model = load_model("model/model.keras")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")

        if not file or file.filename == "":
            return "No file uploaded"

        print("File received:", file.filename)

        # Read image
        file_bytes = file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return "Invalid image file"

        # Resize (IMPORTANT: must match training size)
        img = cv2.resize(img, (64, 64))

        # Normalize
        img = img.astype("float32") / 255.0

        # Reshape
        img = np.expand_dims(img, axis=0)

        # 🔥 Correct prediction
        prediction = model.predict(img)

        print("RAW PREDICTION:", prediction)

        # 🔥 Handle both binary and multi-class models
        if prediction.shape[1] == 1:
            pred_value = float(prediction[0][0])
            print("Pred value:", pred_value)

            if pred_value > 0.5:
                result = "Suspicious Activity Detected 🚨"
            else:
                result = "Normal Activity ✅"

        else:
            class_idx = np.argmax(prediction)
            print("Class index:", class_idx)

            classes = ["Normal Activity ✅", "Suspicious Activity Detected 🚨"]
            result = classes[class_idx]

        return render_template("index.html", prediction=result)

    except Exception as e:
        print("Error:", str(e))
        return "ERROR: " + str(e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
