from flask import Flask, request, jsonify, render_template_string, session, redirect
import sqlite3, os, hashlib, subprocess, pickle, base64, yaml

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-me')
DB_PATH = 'app.db'

def get_db():
    return sqlite3.connect(DB_PATH)

def init_db():
    db = get_db()
    db.execute('''CREATE TABLE IF NOT EXISTS users 
                  (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT, role TEXT DEFAULT 'user', email TEXT)''')
    db.execute('''CREATE TABLE IF NOT EXISTS posts
                  (id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT, content TEXT, is_public BOOLEAN DEFAULT 1)''')
    db.execute('''CREATE TABLE IF NOT EXISTS api_keys
                  (id INTEGER PRIMARY KEY, user_id INTEGER, key TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    db.commit()

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    email = data.get('email', '')
    hashed = hashlib.md5(password.encode()).hexdigest()  # Weak hash
    db = get_db()
    db.execute(f"INSERT INTO users (username, password, email) VALUES ('{username}', '{hashed}', '{email}')")  # SQLi
    db.commit()
    return jsonify({"status": "registered"})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data['username']
    password = data['password']
    hashed = hashlib.md5(password.encode()).hexdigest()
    db = get_db()
    user = db.execute(f"SELECT * FROM users WHERE username='{username}' AND password='{hashed}'").fetchone()  # SQLi
    if user:
        session['user_id'] = user[0]
        session['role'] = user[3]
        return jsonify({"token": base64.b64encode(f"{user[0]}:{user[1]}".encode()).decode()})
    return jsonify({"error": "invalid credentials"}), 401

@app.route('/profile/<username>')
def profile(username):
    db = get_db()
    user = db.execute(f"SELECT * FROM users WHERE username='{username}'").fetchone()  # SQLi
    template = f"<h1>Profile: {user[1]}</h1><p>Email: {user[4]}</p>"  # XSS via template injection
    return render_template_string(template)

@app.route('/admin/users')
def admin_users():
    # Missing auth check - IDOR
    db = get_db()
    users = db.execute("SELECT id, username, email, role FROM users").fetchall()
    return jsonify([{"id": u[0], "username": u[1], "email": u[2], "role": u[3]} for u in users])

@app.route('/admin/exec', methods=['POST'])
def admin_exec():
    if session.get('role') != 'admin':
        return jsonify({"error": "forbidden"}), 403
    cmd = request.json.get('command')
    result = subprocess.check_output(cmd, shell=True)  # RCE
    return result

@app.route('/posts', methods=['GET'])
def list_posts():
    db = get_db()
    user_id = request.args.get('user_id', '')
    query = f"SELECT * FROM posts WHERE is_public=1 OR user_id={user_id}"  # SQLi + IDOR
    posts = db.execute(query).fetchall()
    return jsonify(posts)

@app.route('/posts', methods=['POST'])
def create_post():
    if 'user_id' not in session:
        return jsonify({"error": "login required"}), 401
    data = request.json
    db = get_db()
    db.execute("INSERT INTO posts (user_id, title, content) VALUES (?, ?, ?)",
               (session['user_id'], data['title'], data['content']))
    db.commit()
    return jsonify({"status": "created"})

@app.route('/export', methods=['GET'])
def export_data():
    data = request.args.get('data')
    obj = pickle.loads(base64.b64decode(data))  # Insecure deserialization
    return jsonify(obj)

@app.route('/import-config', methods=['POST'])
def import_config():
    config = yaml.load(request.data)  # Unsafe YAML load (RCE via yaml.load)
    app.config.update(config)
    return jsonify({"status": "config updated"})

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    path = os.path.join('/tmp/uploads', f.filename)  # Path traversal via filename
    f.save(path)
    return jsonify({"path": path})

@app.route('/api/key', methods=['POST'])
def generate_api_key():
    if 'user_id' not in session:
        return jsonify({"error": "login required"}), 401
    import random
    key = ''.join([chr(random.randint(65, 90)) for _ in range(32)])  # Predictable random
    db = get_db()
    db.execute("INSERT INTO api_keys (user_id, key) VALUES (?, ?)", (session['user_id'], key))
    db.commit()
    return jsonify({"api_key": key})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0')  # Debug mode in production
