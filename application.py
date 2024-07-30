from flask import Flask, render_template, request, redirect, url_for, jsonify
import requests
import pickle
import json

app = Flask(__name__)

@app.route('/')
def home():
    data = 'Welcome to the Advertisement project'
    return render_template('getdata.html', data=data)

@app.route('/submit', methods=["GET", "POST"])
def submit():
    result = 0
    data = {}
    with open('model training/EN_model.pickle', 'rb') as file:
        df = pickle.load(file)
    if request.method== 'POST':
        tv = float(request.form["TV"])
        rd = float(request.form["Radio"])
        np = float(request.form["Newspaper"])
        result = df.predict([[tv, rd, np]])[0]  # Getting the first element of the prediction
        data['tv'] = tv
        data['rd'] = rd
        data['np'] = np
        data['result'] = result
    
    # data['predict'] = df.predict([[12,25,2]])

    return redirect(url_for('predict', score=dict(data)))

@app.route('/predict/<score>')
def predict(score):
    return render_template('submit.html', score=score)

if __name__ == '__main__':
    app.run(debug=True)
