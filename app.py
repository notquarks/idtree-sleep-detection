# import imp
from flask import Flask, render_template, request
import pandas as pd
import joblib
from dotenv import load_dotenv


load_dotenv('./.flaskenv')
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == "POST":

        # Unpickle classifier
        clf = joblib.load("clf.pkl")
        # ['Snoring Range', 'Respiration Rate', 'Temp','Limb Movement', 'Blood Oxygen', 'REM', 'Sleep Hour', 'Heart Rate', ]
        # Get values through input bars
        snoring_range = request.form.get('snoring_range')
        respiration_rate = request.form.get('respiration_rate')
        temperature = request.form.get('temperature')
        limb_movement = request.form.get('limb_movement')
        blood_oxygen = request.form.get('blood_oxygen')
        rem = request.form.get('rem')
        sleep_hour = request.form.get('sleep_hour')
        heart_rate = request.form.get('heart_rate')

        # Put inputs to dataframe
        X = pd.DataFrame([[snoring_range, respiration_rate, temperature, limb_movement, blood_oxygen, rem, sleep_hour, heart_rate]], columns=[
                         "Snoring Range", "Respiration Rate", "Temp", "Limb Movement", "Blood Oxygen", "REM", "Sleep Hour", "Heart Rate"])
        # Get prediction
        prediction = clf.predict(X)[0]

    else:
        prediction = ""

    return render_template('index.html', output=prediction)


if __name__ == '__main__':
    app.run()
