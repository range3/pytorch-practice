import torch
from pathlib import Path

MODEL_NAME = 'range3/textgen'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_length_src = 30
max_length_target = 300

batch_size_train = 8
batch_size_valid = 8

epochs = 1000
patience = 20

WORKSPACE_ROOT_DIR = Path(__file__).parent.parent 
NOVEL_DATA_PATH = (WORKSPACE_ROOT_DIR / 'data/novels/narou').resolve()
SENTENCEPIECE_MODEL_DIR = WORKSPACE_ROOT_DIR / 'models/sentencepiece'
