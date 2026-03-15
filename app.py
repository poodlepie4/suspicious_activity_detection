from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model/model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")
