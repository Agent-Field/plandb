import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/login', methods=['POST'])
def login():
    username = request.json['username']
    password = request.json['password']
    # VULNERABILITY: SQL injection
    conn = sqlite3.connect('app.db')
    cursor = conn.execute(f"SELECT * FROM users WHERE username='{username}' AND password='{password}'")
    user = cursor.fetchone()
    if user:
        return jsonify({"token": "hardcoded-secret-token"})  # VULNERABILITY: hardcoded secret
    return jsonify({"error": "invalid"}), 401

@app.route('/users', methods=['GET'])
def list_users():
    # No auth check
    conn = sqlite3.connect('app.db')
    users = conn.execute("SELECT * FROM users").fetchall()
    return jsonify(users)
