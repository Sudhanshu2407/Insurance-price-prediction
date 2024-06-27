from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import pickle
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the model
model_path = "C:\\sudhanshu_projects\\project-task-training-course\\Insurance-price-prediction\\insurance_price_prediction.pkl"
model = pickle.load(open(model_path, "rb"))

# DataFrame to store user credentials and prediction data
users_df = pd.DataFrame(columns=["username", "password"])
predictions_df = pd.DataFrame(columns=["username", "age", "sex", "bmi", "children", "smoker", "region", "charges"])

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    user = users_df[(users_df['username'] == username) & (users_df['password'] == password)]
    if not user.empty or (username == 'admin' and password == '123'):
        session['username'] = username
        return redirect(url_for('predict'))
    else:
        return redirect(url_for('home'))

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    password = request.form['password']
    users_df.loc[len(users_df)] = [username, password]
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])
        
        features = [[age, sex, bmi, children, smoker, region]]
        charges = model.predict(features)[0]

        predictions_df.loc[len(predictions_df)] = [session['username'], age, sex, bmi, children, smoker, region, charges]
        return render_template('prediction.html', charges=charges)
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
