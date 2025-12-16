import torchvision.transforms.v2 as transforms
from config import Transformations, Config

TRANSFORMATIONS = transforms.Compose(
    [
        Transformations.INPUT_TRANSFORMATIONS,
        transforms.Resize(Config.Transformation.SIZE),
        transforms.RandomAffine(
            degrees=Config.Transformation.Affination.DEGREES,
            translate=Config.Transformation.Affination.TRANSLATION,
            scale=Config.Transformation.Affination.SCALE,
        ),
        Transformations.OUTPUT_TRANSFORMATIONS,
        transforms.Normalize(Config.Transformation.MEAN, Config.Transformation.STD),
    ]
)

INFERENCE_TRANSFORMATIONS = transforms.Compose(
    [
        Transformations.INPUT_TRANSFORMATIONS,
        transforms.Resize(Config.Transformation.SIZE),
        Transformations.OUTPUT_TRANSFORMATIONS,
        transforms.Normalize(Config.Transformation.MEAN, Config.Transformation.STD),
    ]
)
