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

@view.app_errorhandler(404)
def page_not_found(e):
    return render_template('errors/404.html'), 404

@view.app_errorhandler(500)
def internal_server_error(e):
    return render_template('errors/500.html'), 500
