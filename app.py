from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
print("Files in root:", os.listdir())
print("Files in model folder:", os.listdir("model"))

# Load CNN model
model = load_model("model/model.keras")
print("Model loaded successfully")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]

        if file.filename == "":
            return "No file uploaded"

        filename = file.filename.lower()

        # ---------------- IMAGE ----------------
        if filename.endswith(('.png', '.jpg', '.jpeg')):

            file_bytes = file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
            if img is None:
                return "Image decode failed"
        
            img = cv2.resize(img, (64,64))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
        
            pred = model.predict(img)
            print("Raw prediction:", pred)
        
            prediction = float(pred[0][0])

        # ---------------- VIDEO ----------------
        elif filename.endswith(('.mp4', '.avi', '.mov')):

            video_path = "temp_video.mp4"
            file.save(video_path)

            cap = cv2.VideoCapture(video_path)

            predictions = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (64,64))
                frame = frame / 255.0
                frame = np.expand_dims(frame, axis=0)

                pred = model.predict(frame)[0][0]
                predictions.append(pred)

            cap.release()
            os.remove(video_path)

            if len(predictions) == 0:
                return "Error processing video"

            prediction = sum(predictions) / len(predictions)

        else:
            return "Unsupported file type"

        # ---------------- RESULT ----------------
        if prediction > 0.5:
            result = "Suspicious Activity Detected"
        else:
            result = "Normal Activity"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
