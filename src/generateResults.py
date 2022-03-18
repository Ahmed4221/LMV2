import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import torch
from constants import *
import sys

from transformers import LayoutLMTokenizer
from transformers import LayoutLMForTokenClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

args = {'local_rank': -1,
        'overwrite_cache': True,
        'data_dir': 'drive/MyDrive/UBIAI_layoutlm/data_UBIAI',
        'model_name_or_path':'microsoft/layoutlm-base-uncased',
        'max_seq_length': 512,
        'model_type': 'layoutlm',}

# class to turn the keys of a dict into attributes (thanks Stackoverflow)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

args = AttrDict(args)

tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_load(PATH, num_labels):


    model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=num_labels)
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess(image_path):

    image = Image.open(image_path)
    image = image.convert("RGB")

    width, height = image.size
    w_scale = 1000 / width
    h_scale = 1000 / height
    ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
    ocr_df = ocr_df.dropna().assign(left_scaled=ocr_df.left * w_scale,
                    width_scaled=ocr_df.width * w_scale,
                    top_scaled=ocr_df.top * h_scale,
                    height_scaled=ocr_df.height * h_scale,
                    right_scaled=lambda x: x.left_scaled + x.width_scaled,
                    bottom_scaled=lambda x: x.top_scaled + x.height_scaled)
    float_cols = ocr_df.select_dtypes('float').columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)

    words = list(ocr_df.text)
    coordinates = ocr_df[['left', 'top', 'width', 'height']]
    actual_boxes = []
    for idx, row in coordinates.iterrows():
      x, y, w, h = tuple(row) # the row comes in (left, top, width, height) format
      actual_box = [x, y, x+w, y+h] # we turn it into (left, top, left+widght, top+height) to get the actual box
      actual_boxes.append(actual_box)
    boxes = []
    for box in actual_boxes:
        boxes.append(normalize_box(box, width, height))
    return image, words, boxes, actual_boxes

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def convert_example_to_features(image, words, boxes, actual_boxes, tokenizer, args, cls_token_box=[0, 0, 0, 0],
                                sep_token_box=[1000, 1000, 1000, 1000],
                                pad_token_box=[0, 0, 0, 0]):
    width, height = image.size


    tokens = []
    token_boxes = []
    actual_bboxes = []  # we use an extra b because actual_boxes is already used
    token_actual_boxes = []
    for word, box, actual_bbox in zip(words, boxes, actual_boxes):
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)
        token_boxes.extend([box] * len(word_tokens))
        actual_bboxes.extend([actual_bbox] * len(word_tokens))
        token_actual_boxes.extend([actual_bbox] * len(word_tokens))
    # Truncation: account for [CLS] and [SEP] with "- 2".
    special_tokens_count = 2
    if len(tokens) > args.max_seq_length - special_tokens_count:
        tokens = tokens[: (args.max_seq_length - special_tokens_count)]
        token_boxes = token_boxes[: (args.max_seq_length - special_tokens_count)]
        actual_bboxes = actual_bboxes[: (args.max_seq_length - special_tokens_count)]
        token_actual_boxes = token_actual_boxes[: (args.max_seq_length - special_tokens_count)]
    # add [SEP] token, with corresponding token boxes and actual boxes
    tokens += [tokenizer.sep_token]
    token_boxes += [sep_token_box]
    actual_bboxes += [[0, 0, width, height]]
    token_actual_boxes += [[0, 0, width, height]]

    segment_ids = [0] * len(tokens)
    # next: [CLS] token
    tokens = [tokenizer.cls_token] + tokens
    token_boxes = [cls_token_box] + token_boxes
    actual_bboxes = [[0, 0, width, height]] + actual_bboxes
    token_actual_boxes = [[0, 0, width, height]] + token_actual_boxes
    segment_ids = [1] + segment_ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    # Zero-pad up to the sequence length.
    padding_length = args.max_seq_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    input_mask += [0] * padding_length
    segment_ids += [tokenizer.pad_token_id] * padding_length
    token_boxes += [pad_token_box] * padding_length
    token_actual_boxes += [pad_token_box] * padding_length
    assert len(input_ids) == args.max_seq_length
    assert len(input_mask) == args.max_seq_length
    assert len(segment_ids) == args.max_seq_length
    # assert len(label_ids) == args.max_seq_length
    assert len(token_boxes) == args.max_seq_length
    assert len(token_actual_boxes) == args.max_seq_length

    return input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes


def convert_to_features(image, words, boxes, actual_boxes, model):
    input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes = convert_example_to_features(image=image,
                                                                                                      words=words,
                                                                                                      boxes=boxes,
                                                                                                      actual_boxes=actual_boxes,
                                                                                                      tokenizer=tokenizer,
                                                                                                      args=args)
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    attention_mask = torch.tensor(input_mask, device=device).unsqueeze(0)
    token_type_ids = torch.tensor(segment_ids, device=device).unsqueeze(0)
    bbox = torch.tensor(token_boxes, device=device).unsqueeze(0)
    #model=model_load(model_path,num_labels)
    print(token_type_ids)
    outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)
    token_predictions = outputs.logits.argmax(-1).squeeze().tolist()  # the predictions are at the token level
    

    word_level_predictions = []  # let's turn them into word level predictions
    final_boxes = []
    for id, token_pred, box in zip(input_ids.squeeze().tolist(), token_predictions, token_actual_boxes):
        if (tokenizer.decode([id]).startswith("##")) or (id in [tokenizer.cls_token_id,
                                                                tokenizer.sep_token_id,
                                                                tokenizer.pad_token_id]):
            # skip prediction + bounding box

            continue
        else:
            word_level_predictions.append(token_pred)
            final_boxes.append(box)
    return word_level_predictions, final_boxes

def iob_to_label(label):
    if label != 'O':
        return label[2:]
    else:
        return ""


def main():
    path_to_image = sys.argv[1]
    image, words, boxes, actual_boxes = preprocess(path_to_image)
    model = torch.load(os.path.join(MODEL_SAVE_PATH,PATH))
    model.eval()
    print(model)
    word_level_predictions, final_boxes=convert_to_features(image, words, boxes, actual_boxes, model)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    label2color = {'VAT_amount_value':'green',
                'grand_total_heading':'green',
                'total_heading':'blue',
                'Extra':'blue',
                'VAT_rate_value':'black',
                'month_heading':'black',
                'price_per_unit_value':'red',
                'discount_heading':'red',
                 'applicable_VAT_percentage_value':'grey',
                 'applicable_VAT_percentage_heading':'grey',
                 'quantity_value':'violet',
                  'type_of_good_value':'orange',
                  'type_of_good_heading':'orange',
                  'amount_exluding_VAT_value':'green',
                'VAT_amount_heading':'green',
                'month_value':'blue',
                'amount_including_VAT_value':'blue',
                'price_per_unit_heading':'black',
                'unit_price_excl':'black',
                '_VAT_value':'red',
                'discount_value':'red',
                 'amount_exluding_VAT_heading':'grey',
                 'VAT_rate_heading':'grey',
                 'quantity_heading':'violet',
                  'amount_including_VAT_heading':'orange',
                  'grand_total_value':'orange',
                    'unit_price_excl':'orange',
                  '_VAT_heading':'orange'}
    labels = list(set(overall_labels))
    label_map = {i: label for i, label in enumerate(labels)}

    for prediction, box in zip(word_level_predictions, final_boxes):
        predicted_label = iob_to_label(label_map[prediction]).lower()
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)

    print(type(image))

if __name__ == "__main__":
    main()