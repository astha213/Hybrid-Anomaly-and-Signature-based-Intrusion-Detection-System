from flask import Flask, render_template, request, redirect, session, url_for, make_response
from flask_session import Session
import mysql.connector
from datetime import timedelta
import urllib.parse
import hashlib

import uuid
app = Flask(__name__)

app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=1)
Session(app)

db = mysql.connector.connect(
    host="localhost",
    port='3306',
    user="root",
    password="mustafaN54",
    database="asaa"
)
cursor = db.cursor(dictionary=True)

@app.before_request
def load_user():
    if 'session_id' in request.args:
        session['session_id'] = request.args['session_id']
    else:
        session['session_id'] = None

    if session['session_id']:
        query = "SELECT * FROM users WHERE session_id = %s"
        cursor.execute(query, (session['session_id'],))
        user = cursor.fetchone()
        if user:
            session['username'] = user['username']
        else:
            session.clear()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']

    password = request.form['password']
    query = "SELECT * FROM users WHERE username = %s AND password = %s"
    cursor.execute(query, (username, password))
    user = cursor.fetchone()
    if user:
        session['username'] = username
        # Generate a unique session ID using UUID
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        # Update the user's session ID in the database
        update_query = "UPDATE users SET session_id = %s WHERE username = %s"
        cursor.execute(update_query, (session_id, username))
        db.commit()
        return redirect(url_for('profile'))
    else:
        return "Invalid username or password"


@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    password = request.form['password']
    session_id = hashlib.sha256(username.encode()).hexdigest()
    query = "INSERT INTO users (username, password, session_id) VALUES (%s, %s, %s)"
    cursor.execute(query, (username, password, session_id))
    db.commit()
    session['username'] = username
    session['session_id'] = session_id
    return redirect(url_for('profile'))

@app.route('/profile')
def profile():
    if 'username' in session:
        username = session['username']
        return render_template('profile.html', username=username)
    return redirect(url_for('index'))

@app.route('/save_profile', methods=['POST'])
def save_profile():
    if 'username' in session:
        username = session['username']
        # Update user profile in the database
        return redirect(url_for('profile'))
    else:
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=8080, debug=True)
