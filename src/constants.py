import os
ALLOWED_EXTENSIONS = ['pdf']

if not os.path.exists('data/'):
    os.mkdir('data/')
RAW_DATA_PATH = 'data/raw_data'
if not os.path.exists(RAW_DATA_PATH):
    os.mkdir(RAW_DATA_PATH)
PROCESSED_DATA_PATH = 'data/processed_data'
if not os.path.exists(PROCESSED_DATA_PATH):
    os.mkdir(PROCESSED_DATA_PATH)
PROCESSED_DATA_JSON_PATH = 'data/processed_json'
if not os.path.exists(PROCESSED_DATA_JSON_PATH):
    os.mkdir(PROCESSED_DATA_JSON_PATH)
DRAW = False

IMG_DIR = 'data/converted_data'