from flask import Flask,render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from io import BytesIO
import base64
from flask_sqlalchemy import SQLAlchemy
from mpl_toolkits import mplot3d
import bcrypt


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'secret_key'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


# Sample image data with related content
images = [
    {
        'url': 'static/builtin/google.jpg',
        'title': 'CEO Of GOOGLE',
        'description': 'Sundar Pichai, born in India and holding degrees from IIT Kharagpur, Stanford, and Wharton, is the CEO of Alphabet Inc. and Google, recognized for his collaborative leadership style and significant contributions to Google’s core products and AI advancements.' },
    {
        'url': 'static/builtin/cook.jpg',
        'title': 'CEO Of APPLE',
        'description': 'Tim Cook, CEO of Apple, has led a period of significant growth and innovation with the introduction of products like the Apple Watch and AirPods, expansion of services such as Apple Music and Apple TV+, and a strong commitment to user privacy, renewable energy, and ethical supply chains.'
    },
    {
        'url': 'static/builtin/nividia.jpg',
        'title': 'CEO Of NIVIDIA',
        'description':  'Jensen Huang, co-founder and CEO of NVIDIA, revolutionized the tech industry with GPU innovations, propelling advancements in AI, gaming, and scientific research, and positioning NVIDIA as a leader in autonomous driving, deep learning, and data centers.'}
]

@app.route('/contact')
def contact():
    return render_template('contact.html', images=images)



class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self,email,password,name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))

with app.app_context():
    db.create_all()
    

@app.route('/user')
def user():
    return render_template('user.html')


@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name,email=email,password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')



    return render_template('register.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            return render_template('login.html',error='Invalid user')

    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('indes.html',user=user)
    
    return redirect('/login')

@app.route('/logout')
def logout():
    session.pop('email',None)
    return redirect('/login')

@app.route("/signup")
def sign_up():
    return render_template("signup.html")


# Helper function to preprocess data and train models
def preprocess_and_train(file_path):
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Assuming the dataset has columns 'features' and 'target'
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    
    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    # Train XGBoost model
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    
    # Calculate accuracy and F1 score
    accuracy_rf = accuracy_score(y_test, y_pred_rf) * 100
    f1_rf = f1_score(y_test, y_pred_rf, average='weighted') * 100
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb) * 100
    f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted') * 100
    
    return accuracy_rf, f1_rf, accuracy_xgb, f1_xgb

# Helper function to create 3D pie chart
def create_pie_chart():
    data = {
        'Algorithm': ['Random Forest', 'XGBoost'],
        'Accuracy': [85, 90],
        'F1 Score': [80, 88]
    }
    df = pd.DataFrame(data)
    fig = px.pie(df, values='Accuracy', names='Algorithm', title='Accuracy by Algorithm')
    return fig.to_html()

# Route to handle file upload and processing
@app.route('/indes', methods=['GET', 'POST'])
def indes():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        
        # Save the file to the server
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        # Preprocess the data and get results
        accuracy_rf, f1_rf, accuracy_xgb, f1_xgb = preprocess_and_train(file_path)
        
        # Create visualizations
        pie_chart = create_pie_chart()
        
        # Render the results in the HTML page
        return render_template('analysis.html', 
                            accuracy_rf=accuracy_rf, 
                            f1_rf=f1_rf,
                            accuracy_xgb=accuracy_xgb, 
                            f1_xgb=f1_xgb,
                            pie_chart=pie_chart,
                            rf_status='Yes' if accuracy_rf > 80 else 'No',
                            xgb_status='Yes' if accuracy_xgb > 80 else 'No')

    return render_template('indes.html')


if (__name__) == ("__main__"):
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run (debug=True,host='0.0.0.0')

