from flask import Flask, jsonify, request, redirect, render_template, url_for, flash, session
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
import os
from seg import grader


# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'jpeg'}

db = SQLAlchemy()
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ProfessorSecret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:''@localhost/tobacco_db'

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)


#from .models import User
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(100))
    role = db.Column(db.Integer)

    def __init__(self, email, password, name, role):
        self.email=email
        self.password=password
        self.name=name
        self.role=role

class Tobacco(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    leaf = db.Column(db.String(100))
    grade = db.Column(db.String(50))
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.datetime.now())

    def __init__(self, user_id, leaf, grade, created_at):
        self.user_id=user_id
        self.leaf=leaf
        self.grade=grade
        self.created_at=created_at

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.filter_by(id=user_id).first()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email == '' or password == '':
            flash('some fields are empty.')
            return redirect(url_for('login'))

        user = User.query.filter_by(email=email).first()
        
        if not user:
            flash('Invalid login details.')
            return redirect(url_for('login'))
        if check_password_hash(user.password, password):
            login_user(user)
            session['userid'] = user.id
            return redirect(url_for('dashboard'))

        flash('Invalid login details.')
        return redirect(url_for('login'))

    # If no valid image file was uploaded, show the file upload form:
    return render_template('login.html')


@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == "POST":
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        password2 = request.form.get('password_confirmation')

        if password != password2:
            flash('Password confirmation should match!')
            return redirect(url_for('register'))

        if len(password) <= 7:
            flash('Password should be 8 characters or greater!')
            return redirect(url_for('register'))

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists!')
            return redirect(url_for('register'))
        new_user = User(email=email, password=generate_password_hash(password, method='sha256'), name=name, role=1)
        db.session.add(new_user)
        db.session.commit()

        flash('Successfully register new user!')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/grade', methods=['POST'])
@login_required
def grade():
    if "file1" not in request.files:
            flash('Image was not found!')
            return redirect(url_for('dashboard'))

    file1 = request.files["file1"]
    ct = datetime.datetime.now()

    extension = file1.filename.split('.')[1]
    path = os.path.join("static/graded", str(ct.timestamp()) +"."+ extension)
    file1.save(path)

    graded = grader(path=path)
    new_tobacco = Tobacco(user_id=session['userid'], leaf=path, grade=graded, created_at=ct)
    db.session.add(new_tobacco)

    db.session.commit()
    flash('Image graded as!'+graded)
    return redirect(url_for('dashboard'))


@app.route('/statistics', methods=['GET'])
@login_required
def statistics():
    grades = Tobacco.query.all()
    return render_template('statistics.html', grades=grades)

@app.route('/logout', methods=['GET'])
@login_required
def logout():
    logout_user()
    g=None
    return redirect(url_for('index'))



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)