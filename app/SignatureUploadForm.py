from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField


class SignatureUploadForm(FlaskForm):
    anchor = FileField(
        "Anchor",
        validators=[
            FileRequired(),
            FileAllowed(["jpg", "png", "jpeg"], "Images only!"),
        ],
    )

    sample = FileField(
        "Sample",
        validators=[
            FileRequired(),
            FileAllowed(["jpg", "png", "jpeg"], "Images only!"),
        ],
    )

    submit = SubmitField("Verify Signatures")
