from transformers import AutoModel
import torch

from constants import *


# model = torch.load(TRAINED_MODEL)

# model.save_pretrained(PATH)

model = AutoModel.from_pretrained(MODEL_PATH)