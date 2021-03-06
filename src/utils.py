# Imports
from constants import *
from unittest import skip
from transformers import LayoutLMv2FeatureExtractor
import pandas as pd
import cv2
import numpy as np
import pickle

unique_rows = []    # For row_id function

pkl_word = []
pkl_box = []
pkl_label = []


def unnormalize_box(bbox, width, height, OPTION):
    
    # data = [
        #  int(width * (bbox[0] / 1000)),
        #  int(height * (bbox[1] / 1000)),
        #  int(width * (bbox[2] / 1000)),
        #  int(height * (bbox[3] / 1000)),
    #  ]
    data = [
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3]
    ]
    
    if OPTION=='Quad':
    
    # converting to 8 co-ordinate system (FOR THE CORD!!!)

        quad = {
            "x2": data[2],
            "y3": data[3],
            "x3": data[2],
            "y4": data[3],
            "x1": data[0],
            "y1": data[1],
            "x4": data[0],
            "y2": data[1]
        }

        return quad
    
    else:
        return data


def get_row_id(line, row_id):
    
    if len(unique_rows) == 0:
        id = row_id + 1
        unique_rows.append(
            {
                'row': line,
                'row_id': id
            }
        )

    else:
        found_match = False
        
        for row in unique_rows:    
            
            if abs(line - row['row']) in range(0, 4):

                id = row['row_id']
                found_match = True   
            
            else:
                continue     
        
        if found_match == False:
            id = row_id + 1
            unique_rows.append(
                {
                    'row': line,
                    'row_id': id
                    }
                    )
    return id


def get_data(filename):

    # img = cv2.imread(filename=filename, flags=cv2.COLOR_BGR2RGB)
    # img = filename
    img = cv2.cvtColor(np.array(filename), cv2.COLOR_RGB2BGR)
    height = img.shape[0] 
    width =  img.shape[1]

    feature_extractor = LayoutLMv2FeatureExtractor(do_resize=False)

    encoding = feature_extractor(img, return_tensors="np")
    
    # Removing one layer of wrap
    encoding['words'] = encoding['words'][0]
    encoding['boxes'] = encoding['boxes'][0]    
    
    true_boxes_q = [unnormalize_box(box, width, height, 'Quad') for box in encoding['boxes']]
    true_boxes_c = [unnormalize_box(box, width, height, 'Corner') for box in encoding['boxes']]
    word = []
    row_id = 0

    word_examples = []
    box_examples = []
    label_examples = []

    for box, text, c_box in zip(true_boxes_q, encoding['words'], true_boxes_c):
        
        row_id = get_row_id(box['y3'], row_id)

        word.append({'quad': box,
        'text': text,
        'row_id': row_id,
        'category': 'None',
        'groud_id': 0
        })

        word_examples.append(text)
        box_examples.append(c_box)
        label_examples.append('O')


    df = pd.DataFrame(list(zip(word)), 
    columns=['words'])
    
    pkl_word.append(word_examples)
    pkl_box.append(box_examples)
    pkl_label.append(label_examples)

    return df, img


def create_box(bbox, label, image):

    x1 = bbox['x1']
    y1 = bbox['y1']
    x2 = bbox['x3']
    y2 = bbox['y3']

    image = cv2.rectangle(image, (x1, y1), (x2, y2), (36, 255, 12), 1)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    return image


def get_labelled_image(file):

    data, image = get_data(filename=file)


    for i in range(len(data)):
        
        box, text= data['words'][i]['quad'], data['words'][i]['text']
        # print(data['words'][0]['quad'])
        image = create_box(box, text, image)

    
    return image, data


def draw_image(image):
    
    cv2.imshow('Labelled image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getListOfFiles(dirName,allowed_extensions = ALLOWED_EXTENSIONS):    
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            extenstion = fullPath[-3:]
            if extenstion in allowed_extensions:
                allFiles.append(fullPath)
                
    return allFiles

def save_pickel():
    
    with open('pickel_file.pkl', 'wb') as t:
        pickle.dump([pkl_word, pkl_label, pkl_box], t)