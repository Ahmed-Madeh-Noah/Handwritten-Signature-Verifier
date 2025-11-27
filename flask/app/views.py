from flask import Blueprint

view = Blueprint('view', __name__)

@view.route('/')
def hello():
	return "<H1>Handwritten Signature Verifier</H1>"
