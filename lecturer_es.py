from flask import Flask, render_template, url_for, request, redirect
from datetime import datetime
import json
from random import choice
import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

path = "test.csv"
df = pd.read_csv(path)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('model_pkl' , 'rb') as f:
    model = pickle.load(f)

with open('db.json') as f:
    data = json.load(f)

app = Flask(__name__)

student_info = {
    "Name": "",
    "Matric_Number": "",
    "Year": "",
    "Course": "",
}

curr_question = list(data.keys())[0]
curr_response = data[curr_question]
final_answer = ""


@ app.route('/')
def home():
    messages = [
    "Welcome"
]
    message = choice(messages)

    return render_template("home.html", message=message)


@ app.route("/get-to-know", methods=["GET", "POST"])
def get_info():
    if request.method == "POST":
        if request.form['name'] == '' :
            return redirect("/get-to-know")
        else:
            nlp_text = request.form["name"]
            # ML model goes here
            global prediction
            prediction = detector(nlp_text)
            return redirect("/answer")

    return render_template("information.html")


def detector(text, model=model, vectorizer= vectorizer):
  text = vectorizer.transform([text]).toarray()
  prediction = model.predict(text)

  column_labels = df.columns[1:]
  predicted_labels = [column_labels[i] for i in range(len(column_labels)) if prediction[0, i] == 1]
  return predicted_labels

@app.route('/answer', methods=['GET', 'POST'])
def answer():
    if request.method == 'POST':
        
        global curr_question
        curr_question = "empty_text"
 
        return redirect("/get-to-know")
    
        
    return render_template('answer.html',question=curr_question, prediction=prediction)
    

app.run(host='0.0.0.0', port=5001, debug=True)

