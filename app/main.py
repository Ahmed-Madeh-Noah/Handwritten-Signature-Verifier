from dotenv import load_dotenv
from logging.handlers import SMTPHandler
import os
import logging
from PIL import Image
from flask import Flask, render_template, request
from werkzeug.middleware.proxy_fix import ProxyFix
from SignatureUploadForm import SignatureUploadForm
from io import BytesIO


load_dotenv()

mail_handler = SMTPHandler(
    mailhost=(os.environ["SMTP_HOST_IP"], int(os.environ["SMTP_HOST_PORT"])),
    fromaddr=os.environ["FROM"],
    toaddrs=(os.environ["TO"]),
    subject="Signature Verifier Flask Error",
    credentials=(os.environ["FROM"], os.environ["APP_PASSWORD"]),
    secure=(),
)
mail_handler.setLevel(logging.ERROR)


def call_model(anchor: Image.Image, sample: Image.Image) -> dict[str, bool | int]:
    return {"match": True, "confidence": 98}


app = Flask(__name__)
app.secret_key = os.environ["SECRET_KEY"]
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
if not app.debug:
    app.logger.addHandler(mail_handler)


@app.get("/")
def index():
    return render_template("index.html")


@app.route("/GUI/", methods=["get", "post"])
def gui():
    form = SignatureUploadForm()
    result = None
    if form.validate_on_submit():
        anchor = Image.open(form.anchor.data)
        sample = Image.open(form.sample.data)
        result = call_model(anchor, sample)
    return render_template("gui.html", form=form, result=result)


@app.get("/API/")
def get_api():
    return render_template("api.html")


@app.post("/API/")
def post_api():
    anchor = Image.open(BytesIO(request.files["anchor"].read()))
    sample = Image.open(BytesIO(request.files["sample"].read()))
    return call_model(anchor, sample)


@app.errorhandler(404)
def page_not_found(e):
    return render_template("errors/404.html"), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template("errors/500.html"), 500
