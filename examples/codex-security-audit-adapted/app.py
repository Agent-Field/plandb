from flask import Flask, request, jsonify
import sqlite3, os, subprocess

app = Flask(__name__)
DB = "app.db"
SECRET = "super-secret-key-123"  # hardcoded secret

@app.route('/exec', methods=['POST'])
def run_command():
    cmd = request.json.get('cmd')
    result = subprocess.check_output(cmd, shell=True)  # command injection
    return jsonify({"output": result.decode()})

@app.route('/file', methods=['GET'])
def read_file():
    path = request.args.get('path')
    with open(path) as f:  # path traversal
        return f.read()

@app.route('/search', methods=['GET'])
def search():
    q = request.args.get('q')
    conn = sqlite3.connect(DB)
    results = conn.execute(f"SELECT * FROM items WHERE name LIKE '%{q}%'").fetchall()  # SQLi
    return jsonify(results)
