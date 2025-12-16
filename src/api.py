from SiameseNetwork import SiameseNetwork
from config import Config
import torch
from transformations import INFERENCE_TRANSFORMATIONS
from fastapi import FastAPI, UploadFile
from io import BytesIO
from PIL import Image
import torch.nn.functional as F


MODEL = SiameseNetwork()
MODEL.load_state_dict(torch.load(Config.Model.PATH, weights_only=True))
MODEL.eval()

app = FastAPI()


@app.post("/INFER/")
async def infer(anchor: UploadFile, sample: UploadFile):
    anchor_bytes = await anchor.read()
    sample_bytes = await sample.read()
    anchor_bytes = BytesIO(anchor_bytes)
    sample_bytes = BytesIO(sample_bytes)
    anchor = Image.open(anchor_bytes)
    sample = Image.open(sample_bytes)
    anchor = INFERENCE_TRANSFORMATIONS(anchor)
    sample = INFERENCE_TRANSFORMATIONS(sample)
    anchor_y, sample_y = MODEL(anchor.unsqueeze(0), sample.unsqueeze(0))
    euclidean_distance = F.pairwise_distance(anchor_y, sample_y).item()
    same = euclidean_distance < Config.Model.THRESHOLD
    confidence = min(abs(euclidean_distance - Config.Model.THRESHOLD) / Config.Model.THRESHOLD, 1.0)
    return {
        "same": same,
        "confidence": round(confidence * 100, 2),
    }
