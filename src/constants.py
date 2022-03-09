import os
ALLOWED_EXTENSIONS = ['pdf', 'jpg']

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
DRAW = False

TRAIN_IMG_DIR = 'data/converted_train_data'

TRAINED_MODEL = ''

TEST_IMG_DIR = 'data/converted_test_data'

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