from flask import Flask, request, jsonify, make_response
import sqlite3, jwt, os, xml.etree.ElementTree as ET

app = Flask(__name__)
JWT_SECRET = "changeme123"  # weak secret
DB = "store.db"

def get_db():
    return sqlite3.connect(DB)

def init_db():
    db = get_db()
    db.execute("CREATE TABLE IF NOT EXISTS products (id INTEGER PRIMARY KEY, name TEXT, price REAL, stock INTEGER)")
    db.execute("CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, user_id TEXT, product_id INTEGER, quantity INTEGER)")
    db.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, email TEXT UNIQUE, password TEXT, is_admin BOOLEAN DEFAULT 0)")
    db.commit()

@app.route('/api/products', methods=['GET'])
def search_products():
    q = request.args.get('q', '')
    db = get_db()
    products = db.execute(f"SELECT * FROM products WHERE name LIKE '%{q}%'").fetchall()  # SQLi
    return jsonify(products)

@app.route('/api/products/<int:id>/review', methods=['POST'])
def add_review(id):
    review = request.json.get('text', '')
    return f"<div class='review'>{review}</div>"  # XSS

@app.route('/api/orders', methods=['POST'])
def create_order():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])  # Weak secret + HS256
    except:
        return jsonify({"error": "unauthorized"}), 401
    
    data = request.json
    product_id = data.get('product_id')
    quantity = data.get('quantity', 1)
    # No stock validation — can order negative quantities for refund fraud
    db = get_db()
    db.execute("INSERT INTO orders (user_id, product_id, quantity) VALUES (?, ?, ?)",
               (payload['sub'], product_id, quantity))
    db.commit()
    return jsonify({"status": "ordered"})

@app.route('/api/orders', methods=['GET'])
def list_orders():
    user_id = request.args.get('user_id')  # IDOR — no auth check
    db = get_db()
    orders = db.execute(f"SELECT * FROM orders WHERE user_id='{user_id}'").fetchall()  # SQLi
    return jsonify(orders)

@app.route('/api/import', methods=['POST'])
def import_xml():
    xml_data = request.data
    tree = ET.fromstring(xml_data)  # XXE
    products = []
    for item in tree.findall('.//product'):
        products.append({"name": item.find('name').text, "price": item.find('price').text})
    return jsonify(products)

@app.route('/api/admin/reset', methods=['POST'])
def reset_db():
    # No auth check
    db = get_db()
    db.execute("DELETE FROM orders")
    db.execute("DELETE FROM products")
    db.commit()
    return jsonify({"status": "reset complete"})

@app.route('/api/export', methods=['GET'])
def export_data():
    fmt = request.args.get('format', 'json')
    db = get_db()
    data = db.execute("SELECT * FROM products").fetchall()
    resp = make_response(jsonify(data))
    resp.headers['Access-Control-Allow-Origin'] = '*'  # CORS misconfiguration
    return resp

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
