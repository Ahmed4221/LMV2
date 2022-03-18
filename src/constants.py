import os

"""
This file contains all the config variables for the project
"""
# --------------------  PATHS ------------------------------------------
if not os.path.exists('data/'):
    os.mkdir('data/')
RAW_DATA_PATH = 'data/raw_train_data'
if not os.path.exists(RAW_DATA_PATH):
    os.mkdir(RAW_DATA_PATH)
PROCESSED_DATA_PATH = 'data/processed_train_data'
if not os.path.exists(PROCESSED_DATA_PATH):
    os.mkdir(PROCESSED_DATA_PATH)
PROCESSED_DATA_JSON_PATH = 'data/processed_train_json'
if not os.path.exists(PROCESSED_DATA_JSON_PATH):
    os.mkdir(PROCESSED_DATA_JSON_PATH)
MODEL_SAVE_PATH = 'models/'
if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)
PICKLE_SAVE_PATH = 'pkl_files/'
if not os.path.exists(PICKLE_SAVE_PATH):
    os.mkdir(PICKLE_SAVE_PATH)

TRAIN_IMG_DIR = 'data/converted_train_data'
TEST_IMG_DIR = 'data/converted_test_data'
PATH = "entire_model.pt"

# --------------------  BOOL CONTROLLERS ------------------------------------------
DRAW = False
LABEL_DATA = False
TRAIN = False
TEST = True

# --------------------  CONST VARIABLES ------------------------------------------
TRAINED_MODEL = ''
TRAIN_PICKLE_NAME = 'train_pickle.pkl'
TRAIN_PICKLE_NAME = os.path.join(PICKLE_SAVE_PATH,TRAIN_PICKLE_NAME)
TEST_PICKLE_NAME = 'test_pickle.pkl'
TEST_PICKLE_NAME = os.path.join(PICKLE_SAVE_PATH,TEST_PICKLE_NAME)
ALLOWED_EXTENSIONS = ['pdf', 'jpg']
ground_labels = {'0': 'month_heading',
                '1': 'month_value',
                '2': 'quantity_heading',
                '3': 'quantity_value',
                '4': 'type_of_good_heading',
                '5': 'type_of_good_value',
                '6': 'price_per_unit_heading',
                '7': 'price_per_unit_value',
                '8': 'amount_exluding_VAT_heading',
                '9': 'amount_exluding_VAT_value',
                '10': 'VAT_amount_heading',
                '11': 'VAT_amount_value',
                '12': 'amount_including_VAT_heading',
                '13': 'amount_including_VAT_value',
                '14': 'total_heading',
                '15': 'unit_price_excl._VAT_heading',
                '16': 'unit_price_excl._VAT_value',
                '17': 'total_value',
                '18': 'grand_total',
                '19': 'VAT_rate_heading',
                '20': 'VAT_rate_value',
                '21': 'applicable_VAT_percentage_heading',
                '22': 'applicable_VAT_percentage_value',
                '23': 'discount_heading',
                '24': 'discount_value',
                '25':'Extra'
                }

our_labels = [
    'company name'
    'address'
    'serial code'
    'article code'
    'table heading'
    'invoice heading'
    'invoice code'
    'client code'
    'date heading'
    'date'
    'product heading'
    'product'
    'quantity'
    'price per unit'
    'VAT'
    'total VAT'
    'amount'
    'total amount'
    'contact heading'
    'email'
    'tel'
    'symbol'
    'signature'
    'website'
    'IBAN'
    'discount'
    'no text'
    'bar code'
]

MODEL_CONFIG = {
        'global_step' : 0,
        'num_train_epochs' : 20,
        'lr' : 5e-5
}

# ----------------------------------------------------
overall_labels = ['VAT_amount_value', 'grand_total_heading', 'total_heading', 'Extra', 'VAT_rate_value', 'month_heading', 'total_value', 'price_per_unit_value', 'discount_heading', 'applicable_VAT_percentage_value', 'applicable_VAT_percentage_heading', 'quantity_value', 'type_of_good_value', 'type_of_good_heading', 'amount_exluding_VAT_value', 'VAT_amount_heading', 'month_value', 'amount_including_VAT_value', 'price_per_unit_heading', 'unit_price_excl._VAT_value', 'discount_value', 'amount_exluding_VAT_heading', 'VAT_rate_heading', 'quantity_heading', 'amount_including_VAT_heading', 'grand_total_value', 'unit_price_excl._VAT_heading']
label2id = {label: idx for idx, label in enumerate(overall_labels)}
id2label = {idx: label for idx, label in enumerate(overall_labels)}