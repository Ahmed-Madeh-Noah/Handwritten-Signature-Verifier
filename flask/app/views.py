from flask import Blueprint, render_template, request, jsonify

view = Blueprint('view', __name__)

@view.route('/')
def index():
    return render_template('index.html')

@view.route('/API', methods=['GET', 'POST'])
def api():
    if request.method == 'GET':
        return render_template('api.html')
    
    if request.method == 'POST':
        return jsonify({"match": True, "confidence": 0.98})

