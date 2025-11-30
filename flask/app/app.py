from flask import Flask
from .views import view
from dotenv import load_dotenv
from os import getenv
from werkzeug.middleware.proxy_fix import ProxyFix

load_dotenv()
app = Flask(__name__)
app.register_blueprint(view)
app.config['SECRET_KEY'] = getenv('SECRET_KEY')
app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)
