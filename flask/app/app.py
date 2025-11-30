from flask import Flask
from .views import view
from dotenv import load_dotenv
from os import getenv

load_dotenv()
app = Flask(__name__)
app.register_blueprint(view)
app.config['SECRET_KEY'] = getenv('SECRET_KEY')
