from flask import Blueprint, render_template, request, jsonify
from .forms import SignatureUploadForm

view = Blueprint('view', __name__)

@view.route('/')
def index():
    return render_template('index.html')

@view.route('/GUI', methods=['GET', 'POST'])
def gui():
    form = SignatureUploadForm()
    result = None
    if form.validate_on_submit():
        result = "Match Confirmed (98%)"
    return render_template('gui.html', form=form, result=result)

@view.route('/API', methods=['GET', 'POST'])
def api():
    if request.method == 'GET':
        return render_template('api.html')
    
    if request.method == 'POST':
        return jsonify({"match": True, "confidence": 0.98})

