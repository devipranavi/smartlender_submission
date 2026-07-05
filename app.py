import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, render_template
from xgboost import XGBClassifier

app = Flask(__name__)

# Load the trained model using XGBoost's native format (more portable than pickle
# across different machines/versions) and the scaler using pickle (safe for sklearn objects)
model = XGBClassifier()
model.load_model('rdf_model.json')
scale = pickle.load(open('scale1.pkl', 'rb'))


@app.route('/')  # rendering the home page
def home():
    return render_template('home.html')


@app.route('/predict', methods=["POST", "GET"])  # rendering the input form
def predict():
    return render_template("input.html")


@app.route('/submit', methods=["POST", "GET"])  # route to show the prediction
def submit():
    # Reading the inputs given by the user
    # NOTE: float() is used instead of int() so that income/loan-amount
    # values entered by the applicant aren't accidentally truncated.
    input_feature = [float(x) for x in request.form.values()]
    input_feature = [np.array(input_feature)]

    names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
              'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
              'Loan_Amount_Term', 'Credit_History', 'Property_Area']

    data = pd.DataFrame(input_feature, columns=names)

    # Scale the input the same way the training data was scaled
    data_scaled = scale.transform(data)

    # Predict using the loaded model file
    prediction = model.predict(data_scaled)
    prediction = int(prediction[0])

    if prediction == 0:
        return render_template("output.html", result="Loan Will Not be Approved")
    else:
        return render_template("output.html", result="Loan Will be Approved")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)
