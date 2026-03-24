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
        # ✅ FIX 1: correct method
        file = request.files.get("file")

        if not file or file.filename == "":
            return "No file uploaded"

        print("File received:", file.filename)

        # ✅ Read image safely (no extension check needed)
        file_bytes = file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return "Invalid image file"

        # ✅ FIX 2: correct size
        img = cv2.resize(img, (64, 64))

        # Normalize
        img = img.astype("float32") / 255.0

        # Reshape
        img = np.expand_dims(img, axis=0)

        # ✅ FIX 3: correct prediction
        prediction = model(img, training=False)

        pred_value = float(prediction[0][0])

        if pred_value > 0.5:
            result = "Suspicious Activity Detected 🚨"
        else:
            result = "Normal Activity ✅"

        return render_template("index.html", prediction=result)

    except Exception as e:
        print("Error:", str(e))
        return "ERROR: " + str(e)




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # for Render
    app.run(host="0.0.0.0", port=port)
