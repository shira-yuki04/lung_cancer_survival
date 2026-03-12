from flask import Flask, render_template, request
import joblib
import numpy as np
app = Flask(__name__)
model = joblib.load('lung_cancer_model.pkl')
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods =['POST'])
def predict():
      features = [
        int(request.form['age']),
        int(request.form['gender']),
        int(request.form['country']),
        int(request.form['cancer_stage']),
        int(request.form['family_history']),
        float(request.form['bmi']),
        int(request.form['cholesterol_level']),
        int(request.form['hypertension']),
        int(request.form['asthma']),
        int(request.form['cirrhosis']),
        int(request.form['other_cancer']),
        int(request.form['smoking_status']),
        int(request.form['treatment_type'])
    ]
      prediction = model.predict([features])
      result = 'Survived' if prediction[0] == 1 else 'Not Survived'
      return render_template('index.html', prediction_text='Patient is likely to be {}'.format(result))
if __name__ == '__main__':    app.run(debug=True)