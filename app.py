from flask import Flask, request, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import numpy as np
import config

app = Flask(__name__)

model = load_model('20200530_insurance_xgbreg_deployment_harsha')
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns=cols)
    prediction = predict_model(model, data=data_unseen, round=0)
    prediction = int(prediction.Label[0])
    return render_template('home.html', pred='Expected Bill will be {}'.format(prediction))
    # return render_template('home.html', pred='Expected Bill will be {}'.format("prediction"))
    # print("Prediction Page")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)
    # return jsonify("prediction")
    # print("Prediction API")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)
