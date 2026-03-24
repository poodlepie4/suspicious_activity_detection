from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# 🔥 Reduce CPU usage (important for Render)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB limit

# Debug
print("Files in root:", os.listdir())
print("Files in model folder:", os.listdir("model"))

# Load model
model = load_model("model/model.keras")
print("Model loaded successfully")
print("Model input shape:", model.input_shape)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")

        if not file or file.filename == "":
            return "No file uploaded"

        filename = file.filename.lower()
        print("File received:", filename)

        # ================= IMAGE =================
        if filename.endswith(('.png', '.jpg', '.jpeg', '.webp', '.jfif')):

            file_bytes = file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                return "Invalid image file"

            # Resize to model input
            img = cv2.resize(img, (64, 64))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            # Prediction
            pred = model(img, training=False)
            pred_value = float(pred[0][0])

            print("Prediction value (image):", pred_value)

        # ================= VIDEO =================
        elif filename.endswith(('.mp4', '.avi', '.mov')):

            video_path = "temp_video.mp4"
            file.save(video_path)

            cap = cv2.VideoCapture(video_path)

            predictions = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # 🔥 Process only every 10th frame
                if frame_count % 10 != 0:
                    continue

                frame = cv2.resize(frame, (64, 64))
                frame = frame.astype("float32") / 255.0
                frame = np.expand_dims(frame, axis=0)

                pred = model(frame, training=False)
                predictions.append(float(pred[0][0]))

            cap.release()
            os.remove(video_path)

            if len(predictions) == 0:
                return "Error processing video"

            pred_value = sum(predictions) / len(predictions)

            print("Prediction value (video avg):", pred_value)

        else:
            return "Unsupported file type"

        # ================= RESULT =================
        print("Final prediction value:", pred_value)

        if pred_value > 0.5:
            result = "Suspicious Activity Detected 🚨"
        else:
            result = "Normal Activity ✅"

        return render_template("index.html", prediction=result)

    except Exception as e:
        print("Error:", str(e))
        return "ERROR: " + str(e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
