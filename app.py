from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)

    prediction = model.predict(final_features)

    if prediction[0] == 1:
        result = "Suspicious Activity Detected"
    else:
        result = "Normal Activity"

    return render_template("index.html", prediction=result)




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
