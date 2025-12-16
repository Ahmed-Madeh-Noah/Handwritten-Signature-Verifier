from dotenv import load_dotenv
from logging.handlers import SMTPHandler
import os
import logging
from werkzeug.datastructures.file_storage import FileStorage
import requests
from flask import Flask, render_template, request
from werkzeug.middleware.proxy_fix import ProxyFix
from SignatureUploadForm import SignatureUploadForm


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


def model_inference(anchor: FileStorage, sample: FileStorage) -> dict[str, bool | int]:
    MODEL_ENDPOINT = "http://src:8000/INFER/"
    files_payload = {
        "anchor": (
            anchor.filename or "anchor.png",
            anchor.stream,
            anchor.content_type or "image/png",
        ),
        "sample": (
            sample.filename or "sample.png",
            sample.stream,
            sample.content_type or "image/png",
        ),
    }
    response = requests.post(MODEL_ENDPOINT, files=files_payload)
    response.raise_for_status()
    return response.json()


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
        anchor = form.anchor.data
        sample = form.sample.data
        result = model_inference(anchor, sample)
    return render_template("gui.html", form=form, result=result)


@app.get("/API/")
def get_api():
    return render_template("api.html")


@app.post("/API/")
def post_api():
    return model_inference(request.files["anchor"], request.files["sample"])


@app.errorhandler(404)
def page_not_found(e):
    return render_template("errors/404.html"), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template("errors/500.html"), 500
