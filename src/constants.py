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

ground_labels = {'0': 'name_of_vendor',
                '1': 'quantity_heading',
                '2': 'quantity',
                '3': 'type_of_good_heading',
                '4': 'type_of_good',
                '5': 'price_per_unit_heading',
                '6': 'price_per_unit',
                '7': 'discount_heading',
                '8': 'discount',
                '9': 'net_price_heading',
                '10': 'net_price',
                '11': 'total_heading',
                '12': 'total_amount',
                '13': 'grand_total_heading',
                '14': 'table_heading',
                '15': 'item_name',
                '16': 'extra',
                '17': 'symbol'
                }